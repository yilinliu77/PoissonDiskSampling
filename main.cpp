#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh/Surface_mesh.h>
#include <CGAL/Surface_mesh/IO.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/polygon_mesh_processing.h>
#include <CGAL/IO/OBJ.h>
#include <CGAL/IO/PLY.h>

#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>
#include <boost/iterator/zip_iterator.hpp>
#include <utility>

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <random>
#include <glog/logging.h>

#include <Eigen/Eigen>

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Point_3<K> Point_3;
typedef CGAL::Vector_3<K> Vector_3;
typedef CGAL::Point_2<K> Point_2;
typedef CGAL::Point_set_3<K::Point_3> Point_set;
typedef CGAL::Surface_mesh<K::Point_3> Surface_mesh;

typedef boost::tuple<Point_3, int>                           Point_and_int;
typedef CGAL::Search_traits_3<K>                       Traits_base;
typedef CGAL::Search_traits_adapter<Point_and_int,
    CGAL::Nth_of_tuple_property_map<0, Point_and_int>,
    Traits_base>                                              Traits;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>          K_neighbor_search;
typedef K_neighbor_search::Tree                             Tree;
typedef K_neighbor_search::Distance                         Distance;
typedef CGAL::Fuzzy_sphere<Traits> Fuzzy_sphere;

bool isValidPoint(const Eigen::MatrixXi& grid, const std::vector<Point_2>& points, float cellsize,
    int gwidth, int gheight,
    const Point_2& p, float radius, const int size_of_disk) {
    /* Make sure the point is on the screen */
    if (p.x() < 0 || p.x() >= size_of_disk || p.y() < 0 || p.y() >= size_of_disk)
        return false;

    /* Check neighboring eight cells */
    int xindex = floor(p.x() / cellsize);
    int yindex = floor(p.y() / cellsize);
    int i0 = std::max(xindex - 1, 0);
    int i1 = std::min(xindex + 1, gwidth - 1);
    int j0 = std::max(yindex - 1, 0);
    int j1 = std::min(yindex + 1, gheight - 1);

    for (int i = i0; i <= i1; i++)
        for (int j = j0; j <= j1; j++)
            if (grid(j,i) != -1)
                if (CGAL::squared_distance(points[grid(j, i)],p) < radius * radius)
                    return false;

    /* If we get here, return true */
    return true;
}

int insertPoint(Eigen::MatrixXi& grid, std::vector<Point_2>& points, float cellsize, const Point_2& point) {
    const int idx = points.size();
    points.emplace_back(point);
    int xindex = floor(point.x() / cellsize);
    int yindex = floor(point.y() / cellsize);
    grid(yindex,xindex) = idx;
    return idx;
}


// Poisson sampling in a disk
std::vector<Point_2> poisson_disk_in_triangle(const float radius, const int k)
{
    const int size_of_disk = 500;

    int N = 2;
    /* The final set of points to return */
    std::vector<Point_2> points;
    /* The currently "active" set of points */
    std::vector<int> active;
    /* Initial point p0 */
    std::uniform_int_distribution<int> unif(0, size_of_disk);
    std::mt19937_64 mt(0);

    Point_2 p0(unif(mt), unif(mt));
    float cellsize = floor(radius / sqrt(N));

    /* Figure out no. of cells in the grid for our canvas */
    int ncells_width = ceil(size_of_disk / cellsize) + 1;
    int ncells_height = ceil(size_of_disk / cellsize) + 1;

    /* Allocate the grid an initialize all elements to null */
    Eigen::MatrixXi grid;
    grid.resize(ncells_height, ncells_width);
    grid.fill(-1);

    const int idx = insertPoint(grid, points, cellsize, p0);
    active.emplace_back(idx);

    while (active.size() > 0) 
    {
        std::uniform_int_distribution<int> unif_id(0, active.size()-1);

        int random_index = unif_id(mt);
        const auto& p = points[active[random_index]];

        bool found = false;
        for (int tries = 0; tries < k; tries++) 
        {
            std::uniform_real_distribution<float> unif_theta(0, M_PI * 2);
            std::uniform_real_distribution<float> unif_r(radius, 2 * radius);

            const float theta = unif_theta(mt);
            const float new_radius = unif_r(mt);
            const float pnewx = p.x() + new_radius * cos(theta);
            const float pnewy = p.y() + new_radius * sin(theta);
            const Point_2 pnew(pnewx, pnewy);

            if (!isValidPoint(grid, points, cellsize,
                ncells_width, ncells_height,
                pnew, radius, size_of_disk))
                continue;

            active.emplace_back(insertPoint(grid, points, cellsize, pnew));
            found = true;
            break;
        }

        /* If no point was found after k tries, remove p */
        if (!found)
            active.erase(active.begin() + random_index);
    }

    return points;
}

// Montecarlo sampling in a triangle mesh
// The triangle id of the sampled point is inside a property map called "face_index"
// It also has the corresponding normal map
Point_set sample_points(const Surface_mesh& v_mesh, const int v_num_points)
{
    std::mt19937 gen; std::uniform_real_distribution<double> dist(0.0f, 1.0f);
    Point_set o_point_set(true);
    auto index_map = o_point_set.add_property_map<int>("face_index", 0).first;
    auto normal_map = o_point_set.normal_map();
    double total_area = CGAL::Polygon_mesh_processing::area(v_mesh);
    double point_per_area = (double)v_num_points / total_area;

    #pragma omp parallel for
    for (int i_face = 0; i_face < v_mesh.num_faces(); ++i_face)
    {
        const auto it_face = *(v_mesh.faces_begin() + i_face);
        Point_3 vertexes[3];
        int i_vertex = 0;
        for (auto it_vertex = (v_mesh.vertices_around_face(v_mesh.halfedge(it_face))).begin(); it_vertex != (v_mesh.vertices_around_face(v_mesh.halfedge(it_face))).end(); ++it_vertex)
        {
            vertexes[i_vertex++] = v_mesh.point(*it_vertex);
        }

        Vector_3 normal = CGAL::cross_product(vertexes[1] - vertexes[0], vertexes[2] - vertexes[0]);
        normal /= std::sqrt(normal.squared_length());

        double area = CGAL::Polygon_mesh_processing::face_area(it_face, v_mesh);

        double face_samples = area * point_per_area;
        unsigned int num_face_samples = face_samples;

        if (dist(gen) < (face_samples - static_cast<double>(num_face_samples))) {
            num_face_samples += 1;
        }

        for (unsigned int j = 0; j < num_face_samples; ++j) {
            double r1 = dist(gen);
            double r2 = dist(gen);

            double tmp = std::sqrt(r1);
            double u = 1.0f - tmp;
            double v = r2 * tmp;

            double w = 1.0f - v - u;
            auto point = Point_3(
                u * vertexes[0].x() + v * vertexes[1].x() + w * vertexes[2].x(),
                u * vertexes[0].y() + v * vertexes[1].y() + w * vertexes[2].y(),
                u * vertexes[0].z() + v * vertexes[1].z() + w * vertexes[2].z()
            );
            #pragma omp critical
            {
                const auto it = o_point_set.insert(point, normal);
                index_map[*it] = i_face;
                normal_map[*it] = normal;
            }
        }
    }
    return o_point_set;
}

Point_set poisson_disk_sampling(const Surface_mesh& v_mesh, const float number_of_sample, const int ratio_of_monte_carlo_sampling=30)
{
    const double surface_area = CGAL::Polygon_mesh_processing::area(v_mesh);
    const double radius = std::sqrt(surface_area / number_of_sample) * 0.75f; // Guessed radius by the number of sample

    // Perform montecarle sampling in advance
    // The triangle id of the sampled point is inside a property map called "face_index"
    Point_set montecarlo_sampling = sample_points(v_mesh, number_of_sample * ratio_of_monte_carlo_sampling);
    CGAL::IO::write_point_set("sampled_points_montecarlo.ply", montecarlo_sampling);
    const auto face_map_montecarlo = montecarlo_sampling.property_map<int>("face_index").first;
    LOG(INFO) << "Done montecarlo sampling";

    // Build kdtree for the montecarlo sampling
    std::vector<unsigned int> indices(montecarlo_sampling.begin(), montecarlo_sampling.end());
    std::iota(montecarlo_sampling.begin(), montecarlo_sampling.end(), 0);
    Tree tree(boost::make_zip_iterator(boost::make_tuple(montecarlo_sampling.point_map().begin(), indices.begin())),
        boost::make_zip_iterator(boost::make_tuple(montecarlo_sampling.point_map().end(), indices.end())));

    Point_set out_points(true);
    auto index_map = out_points.add_property_map<int>("index", 0).first;
    auto normal_map = out_points.normal_map();

    std::vector<bool> is_visited(montecarlo_sampling.size(), false);
    std::vector<bool> is_stored(montecarlo_sampling.size(), false);

    std::vector<int> indexes(montecarlo_sampling.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    // Shuffle the point set
    std::random_device rd;
    std::mt19937_64 g(rd());
    std::shuffle(indexes.begin(), indexes.end(), g);
    for(int i_monte: indexes)
    {
        // This sample has already been travelled
	    if (is_visited[i_monte])
		    continue;

        bool is_conflict = false;
        // Check if the sample is valid
        const Point_3& query = montecarlo_sampling.point(i_monte);
        Distance tr_dist;
        std::vector<Point_and_int> searched_points;
        Fuzzy_sphere search_range(query, radius);
        tree.search(std::back_inserter(searched_points), search_range);

        for(int i_neighbour=0;i_neighbour<searched_points.size();++i_neighbour)
        {
	        const auto& neighbour = searched_points[i_neighbour];
			if (is_stored[neighbour.get<1>()] && CGAL::squared_distance(query, neighbour.get<0>()) < radius * radius)
			{
                is_conflict = true;
                break;
			}
		}

        if (!is_conflict)
        {
	        const auto it = out_points.insert(query);
            index_map[*it] = face_map_montecarlo[i_monte];
            normal_map[*it] = montecarlo_sampling.normal(i_monte);

			is_stored[i_monte] = true;
            is_visited[i_monte] = true;
            for (int i_neighbour = 0; i_neighbour < searched_points.size(); ++i_neighbour)
            {
                const auto& neighbour = searched_points[i_neighbour];
                is_visited[neighbour.get<1>()] = true;
            }
		}
    }

    return out_points;
}

int main()
{
	google::InitGoogleLogging("PoissonDiskSampling");
	FLAGS_logtostderr = 1;

    // Input
	std::string mesh_file = "G:/Dataset/img2brep/test_1817/mesh/00001817_0.ply";
	const int number_of_sample_on_triangle_mesh = 10000;


	CGAL::Surface_mesh<K::Point_3> mesh;
	if (!CGAL::IO::read_PLY(mesh_file, mesh))
	{
		LOG(ERROR) << "Error reading mesh";
		return 0;
	}

    // Sample the mesh
	CGAL::Point_set_3<K::Point_3> points;
	CGAL::Polygon_mesh_processing::sample_triangle_mesh(mesh, points.point_back_inserter(), CGAL::parameters::number_of_points_on_faces(1000));

	CGAL::IO::write_point_set("sampled_points.ply", points);

    // Poisson disk sampling on a disk
    const float radius = 10;
    const int k = 30;
    const auto poisson_disk_sampling_on_a_disk = poisson_disk_in_triangle(radius, k);
    // Write out
	{
	    CGAL::Point_set_3<Point_3> points_out;
	    for (const auto& p : poisson_disk_sampling_on_a_disk)
	        points_out.insert(Point_3(p.x(), p.y(), 0));
	    CGAL::IO::write_point_set("C:/repo/cgal_poisson/poisson_disk_sampling_on_a_disk.ply", points_out);
    }

    LOG(INFO) << "Start to do poisson sampling";
    auto poisson_disk_sampled_on_a_mesh = poisson_disk_sampling(mesh, number_of_sample_on_triangle_mesh);
    LOG(INFO) << "End sampling with " << poisson_disk_sampled_on_a_mesh.size() << " points";
    CGAL::IO::write_point_set("sampled_points_poisson.ply", poisson_disk_sampled_on_a_mesh);

	LOG(INFO) << "Done sampling";
	return 0;
}
