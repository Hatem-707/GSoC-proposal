#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <numeric>
#include <random>
#include <vector>

std::vector<int> get_chunk(const std::vector<int> &data,
                           const std::vector<int> &splitters, int p) {
  std::vector<int> send_disp(p, 0);
  std::vector<int> send_count(p, 0);
  for (int i = 1; i < p; i++) {
    auto boundary =
        std::lower_bound(data.begin(), data.end(), splitters[i - 1]);
    send_disp[i] = std::distance(data.begin(), boundary);
    send_count[i - 1] = send_disp[i] - send_disp[i - 1];
  }
  send_count[p - 1] = data.size() - send_disp[p - 1];

  std::vector<int> recv_counts(p, 0);

  MPI_Alltoall(send_count.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               MPI_COMM_WORLD);
  std::vector<int> recv_disp(p, 0);
  std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_disp.begin(),
                      0);


  int total_recv = recv_disp.back() + recv_counts.back();
  std::vector<int> chunk(total_recv);
  MPI_Alltoallv(data.data(), send_count.data(), send_disp.data(), MPI_INT,
                chunk.data(), recv_counts.data(), recv_disp.data(), MPI_INT,
                MPI_COMM_WORLD);

  return chunk;
}

std::vector<int> parallel_merge(const std::vector<int> &local_a,
                                const std::vector<int> &local_b, int rank,
                                int p) {
  std::vector<int> sample_a(p);
  std::vector<int> sample_b(p);

  size_t step_a = local_a.size() / p;
  size_t step_b = local_b.size() / p;

  for (int i = 0; i < p; i++) {
    sample_a[i] = local_a[i * step_a];
    sample_b[i] = local_b[i * step_b];
  }

  std::vector<int> gather_s1;
  std::vector<int> gather_s2;

  if (rank == 0) {
    gather_s1.resize(p * p);
    gather_s2.resize(p * p);
  }

  MPI_Gather(sample_a.data(), p, MPI_INT, gather_s1.data(), p, MPI_INT, 0,
             MPI_COMM_WORLD);
  MPI_Gather(sample_b.data(), p, MPI_INT, gather_s2.data(), p, MPI_INT, 0,
             MPI_COMM_WORLD);

  std::vector<int> splitters(p - 1);

  if (rank == 0) {
    std::vector<int> sorted_boundaries(2 * p * p);
    std::merge(gather_s1.begin(), gather_s1.end(), gather_s2.begin(),
               gather_s2.end(), sorted_boundaries.begin());

    int sample_step = 2 * p;
    for (int i = 0; i < p - 1; i++) {
      splitters[i] = sorted_boundaries[(i + 1) * sample_step];
    }
  }

  MPI_Bcast(splitters.data(), p - 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> chunk_a = get_chunk(local_a, splitters, p);
  std::vector<int> chunk_b = get_chunk(local_b, splitters, p);

  std::vector<int> res(chunk_a.size() + chunk_b.size());
  std::merge(chunk_a.begin(), chunk_a.end(), chunk_b.begin(), chunk_b.end(),
             res.begin());
  std::cout << "Node: " << rank << " has " << res.size()
            << " elements" << std::endl;
//   if (!res.empty()) {
//     std::cout << "Node: " << rank << " first element " << res[0] << std::endl;
//   }
  return res;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int n = 1024*8192;
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


  if (world_size < 2) {
    if (world_rank == 0)
      std::cerr << "Requires at least 2 ranks." << std::endl;
    MPI_Finalize();
    return 0;
  }

  std::vector<int> slice_1(n);
  std::vector<int> slice_2(n);

  if (world_rank == 0) {
    std::vector<int> data_1(world_size * n);
    std::vector<int> data_2(world_size * n);
    std::mt19937 gen(std::random_device{}());
    std::binomial_distribution<> dist(
        2, 0.8);

    for (auto &ele : data_1)
      ele = dist(gen);
    for (auto &ele : data_2)
      ele = dist(gen);

    std::sort(data_1.begin(), data_1.end());
    std::sort(data_2.begin(), data_2.end());

    MPI_Scatter(data_1.data(), n, MPI_INT, slice_1.data(), n, MPI_INT, 0,
                MPI_COMM_WORLD);
    MPI_Scatter(data_2.data(), n, MPI_INT, slice_2.data(), n, MPI_INT, 0,
                MPI_COMM_WORLD);


    auto res = parallel_merge(slice_1, slice_2, world_rank, world_size);

    std::vector<int> sizes(world_size);
    int res_size = res.size();
    MPI_Gather(&res_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);


    std::vector<int> displ(world_size,
                           0); // FIX: Must allocate size before scan
    std::exclusive_scan(sizes.begin(), sizes.end(), displ.begin(), 0);

    std::vector<int> full_data(2 * n * world_size);
    MPI_Gatherv(res.data(), res_size, MPI_INT, full_data.data(), sizes.data(),
                displ.data(), MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> ground_truth(2 * n * world_size);
    std::merge(data_1.begin(), data_1.end(), data_2.begin(), data_2.end(),
               ground_truth.begin());

    if (ground_truth == full_data) {
      std::cout << "SUCCESS!" << std::endl;
    } else {
      std::cout << "FAIL" << std::endl;
    }

  } else {
    // Workers just receive their initial slices
    MPI_Scatter(nullptr, n, MPI_INT, slice_1.data(), n, MPI_INT, 0,
                MPI_COMM_WORLD);
    MPI_Scatter(nullptr, n, MPI_INT, slice_2.data(), n, MPI_INT, 0,
                MPI_COMM_WORLD);

    auto res = parallel_merge(slice_1, slice_2, world_rank, world_size);

    int res_size = res.size();
    MPI_Gather(&res_size, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(res.data(), res_size, MPI_INT, nullptr, nullptr, nullptr,
                MPI_INT, 0, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}