#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <iostream>
#include <string>

typedef OpenMesh::TriMesh_ArrayKernelT<> Mesh;

float compute_cotan_weight(
    OpenMesh::Vec3f a,
    OpenMesh::Vec3f b,
    OpenMesh::Vec3f c
) {
    const auto u = b - a;
    const auto v = c - a;
    const float dot = OpenMesh::dot(u, v);
    const float cross_norm = OpenMesh::cross(u, v).norm();

    if (cross_norm < 1e-6f) {
        return 0.0f;
    }

    return 0.5f * dot / cross_norm;
}

std::pair<float, float> compute_cotan_weight_for_vertices(
    const Mesh &mesh, const Mesh::VertexHandle vhi,
    const Mesh::VertexHandle vhj
) {
    auto he = mesh.find_halfedge(vhi, vhj);

    if (!he.is_valid()) {
        return {0.0f, 0.0f};
    }

    Mesh::Point pi = mesh.point(vhi);
    Mesh::Point pj = mesh.point(vhj);

    float a = 0, b = 0;

    if (!mesh.is_boundary(he)) {
        const Mesh::VertexHandle vh_opp = mesh.to_vertex_handle(mesh.next_halfedge_handle(he));
        const Mesh::Point pk = mesh.point(vh_opp);

        a = compute_cotan_weight(pk, pi, pj);
    }

    const auto he_opp = mesh.opposite_halfedge_handle(he);
    if (!mesh.is_boundary(he_opp)) {
        const Mesh::VertexHandle vh_opp = mesh.to_vertex_handle(mesh.next_halfedge_handle(he_opp));
        const Mesh::Point pl = mesh.point(vh_opp);

        b = compute_cotan_weight(pl, pi, pj);
    }

    return {a, b};
}

std::vector<OpenMesh::VertexHandle> get_vertices_of_triangle_face(Mesh &mesh, const Mesh::FaceHandle fh) {
    std::vector<OpenMesh::VertexHandle> vertices;
    for (auto fv_it = mesh.fv_iter(fh); fv_it.is_valid(); ++fv_it) {
        vertices.push_back(*fv_it);
    }

    if (vertices.size() != 3) throw std::runtime_error("Invalid number of vertices");

    return vertices;
}

bool is_angle_obtuse(
    const OpenMesh::Vec3f &a,
    const OpenMesh::Vec3f &b,
    const OpenMesh::Vec3f &c
) {
    const auto ab = b - a;
    const auto ac = c - a;

    return (OpenMesh::dot(ab, ac)) < 0;
}

bool is_triangle_obtuse(Mesh &mesh, const Mesh::FaceHandle fh) {
    const auto vertices = get_vertices_of_triangle_face(mesh, fh);

    const auto p0 = mesh.point(vertices[0]),
            p1 = mesh.point(vertices[1]),
            p2 = mesh.point(vertices[2]);

    if (
        is_angle_obtuse(p0, p1, p2)
        || is_angle_obtuse(p1, p0, p2)
        || is_angle_obtuse(p2, p0, p1)
    )
        return true;

    return false;
}

int find_vertex_index(
    const std::vector<OpenMesh::VertexHandle> &vertices,
    const Mesh::VertexHandle vh
) {
    for (int i = 0; i < 3; i++) {
        if (vertices[i] == vh) {
            return i;
        }
    }

    return -1;
}

float compute_triangle_area(Mesh &mesh, const Mesh::FaceHandle fh) {
    const auto vertices = get_vertices_of_triangle_face(mesh, fh);

    const auto p0 = mesh.point(vertices[0]),
            p1 = mesh.point(vertices[1]),
            p2 = mesh.point(vertices[2]);

    const auto cross = (p1 - p0) % (p2 - p0); // cross product

    return 0.5f * cross.norm();
}

float compute_mixed_area(Mesh &mesh, const Mesh::FaceHandle fh, const Mesh::VertexHandle vh) {
    const auto vertices = get_vertices_of_triangle_face(mesh, fh);
    const auto idx = find_vertex_index(vertices, vh);
    if (idx == -1) return 0.0;

    const auto vhi = vertices[idx],
            vhj = vertices[(idx + 1) % 3],
            vhk = vertices[(idx + 2) % 3];

    const auto vi = mesh.point(vhi),
            vj = mesh.point(vhj),
            vk = mesh.point(vhk);

    if (is_angle_obtuse(vi, vj, vk)) {
        return 0.5f * compute_triangle_area(mesh, fh);
    }

    if (is_angle_obtuse(vj, vi, vk) || is_angle_obtuse(vk, vi, vj)) {
        return 0.25f * compute_triangle_area(mesh, fh);
    }

    // All angles are acute, use Voronoi area
    float voronoi_area = 0.0f;

    const auto e0 = vj - vk;
    const float norm_e0_sq = e0.sqrnorm();
    const float cot_k = compute_cotan_weight(vk, vj, vi);

    voronoi_area += (1.0f / 8.0f) * cot_k * norm_e0_sq;

    // Edge vi-vk
    const auto edge_ik = vk - vi;
    const float len_ik_sq = edge_ik.sqrnorm();

    // Cotangent of angle at vj (opposite to edge vi-vk)
    const float cot_j = compute_cotan_weight(vj, vi, vk);
    voronoi_area += (1.0f / 8.0f) * cot_j * len_ik_sq;

    return voronoi_area;
}

float A(Mesh &mesh, const Mesh::VertexHandle vh) {
    float total_area = 0.0;

    for (auto vf_it = mesh.vf_iter(vh); vf_it.is_valid(); ++vf_it) {
        const OpenMesh::FaceHandle fh = *vf_it;

        if (is_triangle_obtuse(mesh, fh)) {
            total_area += compute_mixed_area(mesh, fh, vh);
        } else {
            // Use barycentric area
            total_area += compute_triangle_area(mesh, fh) / 3.0f;
        }
    }

    return total_area;
}

OpenMesh::Vec3f compute_uniform_laplace_beltrami(Mesh &mesh, const Mesh::VertexHandle vh) {
    const auto vi = mesh.point(vh);
    OpenMesh::Vec3f sum = {0.0f, 0.0f, 0.0f};
    int neighbor_count = 0;
    for (Mesh::VertexVertexIter vv_it = mesh.vv_iter(vh); vv_it.is_valid(); ++vv_it) {
        neighbor_count++;
        const auto neighbor = *vv_it;
        const auto vj = mesh.point(neighbor);

        sum += (vj - vi);
    }

    return (1.0f / neighbor_count) * sum;
}

OpenMesh::Vec3f compute_cotangential_laplace_beltrami(Mesh &mesh, const Mesh::VertexHandle vh) {
    const auto vi = mesh.point(vh);

    OpenMesh::Vec3f sum = {0.0f, 0.0f, 0.0f};

    for (Mesh::VertexVertexIter vv_it = mesh.vv_iter(vh); vv_it.is_valid(); ++vv_it) {
        const auto neighbor = *vv_it;
        auto vj = mesh.point(neighbor);

        auto [aij, bij] = compute_cotan_weight_for_vertices(mesh, vh, neighbor);

        sum += (aij + bij) * (vj - vi);
    }


    return 1 / (2 * A(mesh, vh)) * sum;
}

void iterative_smoothing(Mesh &mesh, const float lambda) {
    for (Mesh::VertexIter vv_it = mesh.vertices_begin(); vv_it != mesh.vertices_end(); ++vv_it) {
        const int vertex_index = vv_it->idx();
        const auto vh = mesh.vertex_handle(vertex_index);
        auto vi = mesh.point(vh);
        auto new_vi = vi + lambda * compute_uniform_laplace_beltrami(mesh, vh);
        mesh.set_point(vh, new_vi);
    }
}

void cotangential_iterative_smoothing(Mesh &mesh, const float lambda) {
    for (Mesh::VertexIter vv_it = mesh.vertices_begin(); vv_it != mesh.vertices_end(); ++vv_it) {
        const int vertex_index = vv_it->idx();
        const auto vh = mesh.vertex_handle(vertex_index);
        auto vi = mesh.point(vh);
        auto new_vi = vi + lambda * compute_cotangential_laplace_beltrami(mesh, vh);
        mesh.set_point(vh, new_vi);
    }
}

std::vector<std::vector<float> > laplace_beltrami_matrix(Mesh &mesh) {
    const auto n_vertices = mesh.n_vertices();
    std::vector<std::vector<float> > M(n_vertices, std::vector<float>(n_vertices, 0.0f));

    // Compute M matrix
    for (auto vh: mesh.vertices()) {
        int i = vh.idx();

        float w = 0.0f;

        for (auto vv_it = mesh.vv_iter(vh); vv_it.is_valid(); ++vv_it) {
            int j = vv_it->idx();

            auto [aij, bij] = compute_cotan_weight_for_vertices(mesh, vh, *vv_it);

            M[i][j] = aij + bij;
            w += aij + bij;
        }

        M[i][i] = -w;
    }

    std::vector<float> diagonal_D(n_vertices, 0.0f);

    // Compute D matrix
    for (int i = 0; i < n_vertices; ++i) {
        diagonal_D[i] = 1.0f / (2.0f * A(mesh, mesh.vertex_handle(i)));
    }

    // Matrix Multiplication L = D * M
    for (int i = 0; i < n_vertices; ++i) {
        for (int j = 0; j < n_vertices; ++j) {
            M[i][j] = diagonal_D[i] * M[i][j];
        }
    }

    return M;
}

// Simple matrix multiplication for square matrices
std::vector<std::vector<float> > mat_mul(
    const std::vector<std::vector<float> > &A,
    const std::vector<std::vector<float> > &B
) {
    const int n = A.size();
    std::vector<std::vector<float> > C(n, std::vector<float>(n, 0.0f));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

void laplace_beltrami_matrix_iterative_smoothing(Mesh &mesh, const float lambda) {
    const auto L = laplace_beltrami_matrix(mesh);
    const auto L2 = mat_mul(L, L);

    // Update vertex positions
    for (auto vh: mesh.vertices()) {
        int i = vh.idx();
        auto vi = mesh.point(vh);
        OpenMesh::Vec3f delta = {0.0f, 0.0f, 0.0f};
        for (int j = 0; j < mesh.n_vertices(); ++j) {
            auto vj = mesh.point(mesh.vertex_handle(j));
            delta += L2[i][j] * (vj - vi);
        }
        auto new_vi = vi + lambda * delta;
        mesh.set_point(vh, new_vi);
    }
}

enum Algorithm {
    UNIFORM_LAPLACE_BELTRAMI,
    COTANGENTIAL_LAPLACE_BELTRAMI,
    LAPLACE_BELTRAMI_MATRIX
};

std::string algorithm_to_string(const Algorithm algo) {
    switch (algo) {
        case UNIFORM_LAPLACE_BELTRAMI:
            return "uniform_laplace_beltrami";
        case COTANGENTIAL_LAPLACE_BELTRAMI:
            return "cotangential_laplace_beltrami";
        case LAPLACE_BELTRAMI_MATRIX:
            return "laplace_beltrami_matrix";
    }
    return "unknown";
}

std::string get_file_base_name(const std::string &filepath) {
    const size_t last_slash = filepath.find_last_of("/\\");
    size_t last_dot = filepath.find_last_of('.');

    if (last_dot == std::string::npos || last_dot < last_slash) {
        last_dot = filepath.length();
    }

    return filepath.substr(last_slash + 1, last_dot - last_slash - 1);
}

std::string get_output_file_name(
    const std::string &base_name,
    const Algorithm algo,
    const int iterations
) {
    return "outputs/" + base_name + "_" + algorithm_to_string(algo) + "_iter_" + std::to_string(iterations) + ".obj";
}

int main(const int argc, char *argv[]) {
    Mesh mesh;

    std::string filename = "models/noisyBunnyLowPoly.obj";
    std::cout << "Loading mesh from: " << filename << std::endl;

    if (argc > 1) {
        filename = argv[1];
    }

    Algorithm algo = UNIFORM_LAPLACE_BELTRAMI;

    if (!OpenMesh::IO::read_mesh(mesh, filename)) {
        std::cerr << "Error: Cannot read mesh from " << filename << std::endl;
        return 1;
    }

    std::cout << "Successfully loaded mesh!" << std::endl;


    constexpr auto iter_smoothing_ratio = 0.1f;
    constexpr auto iterations = 10;

    auto alg_func = [&](Mesh &mesh, const float lambda) {
        Mesh copy = mesh;

        switch (algo) {
            case UNIFORM_LAPLACE_BELTRAMI:
                iterative_smoothing(copy, lambda);
                break;
            case COTANGENTIAL_LAPLACE_BELTRAMI:
                cotangential_iterative_smoothing(copy, lambda);
                break;
            case LAPLACE_BELTRAMI_MATRIX:
                laplace_beltrami_matrix_iterative_smoothing(copy, lambda);
                break;
        }
    };

    for (int i = 0; i < iterations; ++i) {
        std::cout << "Iteration " << (i + 1) << " / " << iterations
                << " using algorithm: " << algorithm_to_string(algo) << std::endl;
        alg_func(mesh, iter_smoothing_ratio);
    }

    OpenMesh::IO::write_mesh(
        mesh,
        get_output_file_name(
            get_file_base_name(filename),
            algo, iterations
        )
    );


    return 0;
}
