#include <iostream>
#include <unordered_set>
#include <filesystem>
#include <bvh/triangle.hpp>
#include <bvh/bvh.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/primitive_intersectors.hpp>
#include <bvh/single_ray_traverser.hpp>
#include "happly.h"

using Vector3 = bvh::Vector3<float>;
using Triangle = bvh::Triangle<float>;
using Bvh = bvh::Bvh<float>;
using Ray = bvh::Ray<float>;
using TraverserHigh = bvh::SingleRayTraverser<Bvh, 64, bvh::MPNodeIntersector<Bvh, 23, 8>>;
using TraverserLow = bvh::SingleRayTraverser<Bvh, 64, bvh::MPNodeIntersector<Bvh, 7, 8>>;
using PrimitiveIntersector = bvh::ClosestPrimitiveIntersector<Bvh, Triangle>;

int main() {
    happly::PLYData ply_data("model.ply");
    std::vector<std::array<double, 3>> v_pos = ply_data.getVertexPositions();
    std::vector<std::vector<size_t>> f_idx = ply_data.getFaceIndices<size_t>();

    std::vector<Triangle> triangles;
    for (auto &face : f_idx) {
        triangles.emplace_back(Vector3(v_pos[face[0]][0], v_pos[face[0]][1], v_pos[face[0]][2]),
                               Vector3(v_pos[face[1]][0], v_pos[face[1]][1], v_pos[face[1]][2]),
                               Vector3(v_pos[face[2]][0], v_pos[face[2]][1], v_pos[face[2]][2]));
    }

    Bvh bvh;
    auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(triangles.data(), triangles.size());
    auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), triangles.size());
    std::cout << "global bounding box: ("
              << global_bbox.min[0] << ", " << global_bbox.min[1] << ", " << global_bbox.min[2] << "), ("
              << global_bbox.max[0] << ", " << global_bbox.max[1] << ", " << global_bbox.max[2] << ")" << std::endl;

    bvh::SweepSahBuilder<Bvh> builder(bvh);
    builder.build(global_bbox, bboxes.get(), centers.get(), triangles.size());

    std::vector<std::pair<size_t, size_t>> edges;
    std::queue<size_t> queue;
    queue.push(0);
    while (!queue.empty()) {
        size_t curr = queue.front();
        queue.pop();
        if (!bvh.nodes[curr].is_leaf()) {
            size_t left_idx = bvh.nodes[curr].first_child_or_primitive;
            size_t right_idx = left_idx + 1;
            edges.emplace_back(curr, left_idx);
            edges.emplace_back(curr, right_idx);
            queue.push(left_idx);
            queue.push(right_idx);
        }
    }

    PrimitiveIntersector primitive_intersector(bvh, triangles.data());
    TraverserHigh traverser_high(bvh);
    TraverserLow traverser_low(bvh);
    std::ifstream ray_queries_file("ray_queries.bin", std::ios::in | std::ios::binary);
    std::filesystem::create_directory("graphs");
    float r[7];
    for (int i = 0; ray_queries_file.read(reinterpret_cast<char*>(&r), 7 * sizeof(float)); i++) {
        if (rand() % 10000 != 0) continue;

        Ray ray(
            Vector3(r[0], r[1], r[2]),
            Vector3(r[3], r[4], r[5]),
            0.f,
            r[6]
        );

        TraverserHigh::Statistics statistics_high;
        traverser_high.traverse(ray, primitive_intersector, statistics_high);

        TraverserLow::Statistics statistics_low;
        traverser_low.traverse(ray, primitive_intersector, statistics_low);

        std::string filepath = "graphs/bvh_" + std::to_string(i) + ".dot";
        std::ofstream dot_file(filepath);
        dot_file << "digraph bvh {";
        dot_file << "\n    node [shape=point]";
        dot_file << "\n    edge [arrowhead=none]";

        for (auto &edge : edges) {
            if (statistics_low.traversed.count(edge.second)) {
                dot_file << "\n    " << edge.first << " -> " << edge.second;
                if (statistics_high.traversed.count(edge.second))  dot_file << " [color=red]";
            }
        }

        dot_file << "\n}";
        dot_file.close();
    }
}
