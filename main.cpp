#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <iostream>
#include <string>

typedef OpenMesh::TriMesh_ArrayKernelT<> Mesh;

float A(const Mesh &mesh, const Mesh::VertexHandle vh) {
    return 1.0f;
}

std::vector<std::vector<float> > laplace_beltrami_matrix(const Mesh &mesh) {
    auto M = std::vector<std::vector<float> >{};
    auto D = std::vector<float>{};
    const auto n_vertices = mesh.n_vertices();
    M.reserve(n_vertices);

    for (int i = 0; i < n_vertices; ++i) {
        M.emplace_back();
        M[i].reserve(n_vertices);
        for (int j = 0; j < n_vertices; ++j) {
        }
    }
}

std::pair<float, float>
cotan_ab_at_vertex(const Mesh &mesh, const Mesh::VertexHandle vhi, const Mesh::VertexHandle vhj) {
    auto he = mesh.find_halfedge(vhi, vhj);

    if (!he.is_valid()) {
        return {0.0f, 0.0f};
    }

    Mesh::Point pi = mesh.point(vhi);
    Mesh::Point pj = mesh.point(vhj);

    float a = 0, b = 0;

    if (!mesh.is_boundary(he)) {
        Mesh::VertexHandle vh_opp = mesh.to_vertex_handle(mesh.next_halfedge_handle(he));
        Mesh::Point pk = mesh.point(vh_opp);

        // Vectors from opposite vertex
        Mesh::Point u = pi - pk;
        Mesh::Point v = pj - pk;
        float dot = OpenMesh::dot(u, v);
        float cross_norm = OpenMesh::cross(u, v).norm();

        if (cross_norm > 1e-6f) {
            a = 0.5f * dot / cross_norm;
        }
    }

    const Mesh::HalfedgeHandle he_opp = mesh.opposite_halfedge_handle(he);
    if (!mesh.is_boundary(he_opp)) {
        Mesh::VertexHandle vh_opp = mesh.to_vertex_handle(mesh.next_halfedge_handle(he_opp));
        Mesh::Point pl = mesh.point(vh_opp);

        // Vectors from opposite vertex
        Mesh::Point u = pi - pl;
        Mesh::Point v = pj - pl;

        float dot = OpenMesh::dot(u, v);
        float cross_norm = OpenMesh::cross(u, v).norm();
        if (cross_norm > 1e-6f) {
            b += 0.5f * dot / cross_norm;
        }
    }

    return {a, b};
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

        auto [aij, bij] = cotan_ab_at_vertex(mesh, vh, neighbor);

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

int main(const int argc, char *argv[]) {
    Mesh mesh;

    std::string filename = "models/noisyBunnyLowPoly.obj";

    if (argc > 1) {
        filename = argv[1];
    }

    std::cout << "Loading mesh from: " << filename << std::endl;

    if (!OpenMesh::IO::read_mesh(mesh, filename)) {
        std::cerr << "Error: Cannot read mesh from " << filename << std::endl;
        return 1;
    }

    std::cout << "Successfully loaded mesh!" << std::endl;

    constexpr auto iter_smoothing_ratio = 0.1f;
    constexpr auto iterations = 10;

    for (int i = 0; i < iterations; ++i)
        iterative_smoothing(mesh, iter_smoothing_ratio);

    OpenMesh::IO::write_mesh(mesh, "outputs/noisyBunnyLowPoly_iterative_smoothing.obj");

    if (!OpenMesh::IO::read_mesh(mesh, filename)) {
        std::cerr << "Error: Cannot read mesh from " << filename << std::endl;
        return 1;
    }

    for (int i = 0; i < iterations; ++i)
        cotangential_iterative_smoothing(mesh, iter_smoothing_ratio);


    OpenMesh::IO::write_mesh(mesh, "outputs/noisyBunnyLowPoly_contangential_iterative_smoothing.obj");

    return 0;
}
