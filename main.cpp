#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

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

/*float A(Mesh &mesh, const Mesh::VertexHandle vh) {
    return 1.0f;
} */

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

std::vector<std::vector<float> > laplace_beltrami_matrix(Mesh &mesh) {
    auto L = std::vector<std::vector<float> >{};
    const auto n_vertices = mesh.n_vertices();
    L.reserve(n_vertices);

    for (int i = 0; i < n_vertices; ++i)
    {
        L.emplace_back();
        L[i].reserve(n_vertices);
        for (int j = 0; j < n_vertices; ++j) 
        {
            L[i].emplace_back(0.0f);
        }
    }

    for (Mesh::VertexIter vv_it_i = mesh.vertices_begin(); vv_it_i != mesh.vertices_end(); ++vv_it_i) {
        const int i = vv_it_i->idx();
        const auto vi = mesh.vertex_handle(i);
        float diag_sum = 0.0f;
        for (Mesh::VertexVertexIter vv_it_j = mesh.vv_iter(vi); vv_it_j.is_valid(); ++vv_it_j)
        {
            const int j = vv_it_j->idx();
            const auto vj = mesh.vertex_handle(j);

            auto [aij, bij] = cotan_ab_at_vertex(mesh, vi, vj);

            L[i][j] = -(aij + bij) / (2 * A(mesh, vi));
            diag_sum -= (L[i][j]);
        }

        L[i][i] = diag_sum;
    }
    
    return L;
}

std::vector<std::vector<float>> laplace_beltrami_matrix_squared(Mesh &mesh) {
    auto L = laplace_beltrami_matrix(mesh);
    const auto n = L.size();
    
    // Initialiser la matrice résultat L² avec des zéros
    std::vector<std::vector<float>> L2(n, std::vector<float>(n, 0.0f));
    
    // Multiplication matricielle L × L
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                L2[i][j] += L[i][k] * L[k][j];
            }
        }
    }
    
    return L2;
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

void square_matrix_iterative_smoothing(Mesh &mesh, const float lambda)
{
    // Variante optimiser local : pas de construction de L ni L² explicite
    // L²*v[i] = L(L*v)[i] = Σ_{j voisin i} w_ij ( (L*v)[j] - (L*v)[i] )

    const auto n = mesh.n_vertices();
    std::vector<OpenMesh::Vec3f> lap1(n, OpenMesh::Vec3f(0,0,0));

    // Premier passage : Laplacien cotangent
    for (Mesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
        const int i = v_it->idx();
        const auto vhi = mesh.vertex_handle(i);
        const auto vi = mesh.point(vhi);
        OpenMesh::Vec3f acc(0,0,0);
        for (Mesh::VertexVertexIter vv_it = mesh.vv_iter(vhi); vv_it.is_valid(); ++vv_it) {
            const int j = vv_it->idx();
            const auto vhj = mesh.vertex_handle(j);
            const auto vj = mesh.point(vhj);
            auto [aij, bij] = cotan_ab_at_vertex(mesh, vhi, vhj);
            float w = -(aij + bij) / (2 * A(mesh, vhi)); // w_ij
            acc += w * (vj - vi);
        }
        lap1[i] = acc; // ceci correspond à L*v[i]
    }

    // Second passage : appliquer à lap1 pour approximer L²*v
    std::vector<OpenMesh::Vec3f> lap2(n, OpenMesh::Vec3f(0,0,0));
    for (Mesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
        const int i = v_it->idx();
        const auto vhi = mesh.vertex_handle(i);
        OpenMesh::Vec3f acc(0,0,0);
        for (Mesh::VertexVertexIter vv_it = mesh.vv_iter(vhi); vv_it.is_valid(); ++vv_it) {
            const int j = vv_it->idx();
            const auto vhj = mesh.vertex_handle(j);
            auto [aij, bij] = cotan_ab_at_vertex(mesh, vhi, vhj);
            float w = -(aij + bij) / (2 * A(mesh, vhi));
            acc += w * (lap1[j] - lap1[i]);
        }
        lap2[i] = acc; // L²*v[i]
    }

    // Mise à jour biharmonique explicite : v_{t+1} = v_t - lambda * L²*v_t
    // Pour stabilité, lambda devrait être très petit (ex: 1e-4 à 1e-3)
    for (int i = 0; i < n; ++i) {
        auto vh = mesh.vertex_handle(i);
        auto vi = mesh.point(vh);
        mesh.set_point(vh, vi - lambda * lap2[i]);
    }
}

// Variante "bis" : on calcule la matrice L puis L² explicitement
// Assez long et gourmand en mémoire.
void square_matrix_iterative_smoothing_bis(Mesh &mesh, const float lambda)
{
    auto L2 = laplace_beltrami_matrix_squared(mesh); // construit L puis L^2
    const auto n = mesh.n_vertices();

    std::vector<OpenMesh::Vec3f> new_positions(n);

    for (Mesh::VertexIter vv_it = mesh.vertices_begin(); vv_it != mesh.vertices_end(); ++vv_it)
    {
        const int i = vv_it->idx();
        const auto vh = mesh.vertex_handle(i);
        const auto vi = mesh.point(vh);

        OpenMesh::Vec3f L2_vi(0.0f, 0.0f, 0.0f);
        for (int j = 0; j < n; ++j) {
            auto vhj = mesh.vertex_handle(j);
            auto vj = mesh.point(vhj);
            L2_vi += L2[i][j] * vj; // (L^2 v)_i
        }

        // Mise à jour stable (biharmonique): v_i - lambda * L^2 v_i
        new_positions[i] = vi - lambda * L2_vi;
    }

    for (int i = 0; i < n; ++i) {
        auto vh = mesh.vertex_handle(i);
        mesh.set_point(vh, new_positions[i]);
    }
}

int main(const int argc, char *argv[]) {
    Mesh mesh;

    std::string filename = "models/bunnyLowPoly.obj";

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

    // Ensure outputs directory exists
    mkdir("outputs", 0777);

    OpenMesh::IO::write_mesh(mesh, "outputs/noisyBunnyLowPoly_iterative_smoothing.obj");

    if (!OpenMesh::IO::read_mesh(mesh, filename)) {
        std::cerr << "Error: Cannot read mesh from " << filename << std::endl;
        return 1;
    }

    for (int i = 0; i < iterations; ++i)
        cotangential_iterative_smoothing(mesh, iter_smoothing_ratio);

    OpenMesh::IO::write_mesh(mesh, "outputs/noisyBunnyLowPoly_contangential_iterative_smoothing.obj");

    // Recharger le maillage original avant le lissage avec L^2
    if (!OpenMesh::IO::read_mesh(mesh, filename)) {
        std::cerr << "Error: Cannot read mesh from " << filename << std::endl;
        return 1;
    }

    for (int i = 0; i < iterations; ++i)
        square_matrix_iterative_smoothing(mesh, iter_smoothing_ratio);

    OpenMesh::IO::write_mesh(mesh, "outputs/noisyBunnyLowPoly_square_matrix_iterative_smoothing.obj");

    return 0;
}
