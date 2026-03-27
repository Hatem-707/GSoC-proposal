#include <algorithm>
#include <cassert>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

struct WideSample {
  int value;
  int index;
  int node_rank;
  WideSample() : value(0), index(0), node_rank(0) {}
  WideSample(int v, int i, int n) : value(v), index(i), node_rank(n) {}
};

std::pair<int, int> coarse_co_rank(const std::vector<WideSample> &samples_a,
                                   const std::vector<WideSample> &samples_b,
                                   int grank, int sample_size) {
  int crank = grank / sample_size; // coarse rank
  int low = (crank > (int)samples_b.size()) ? crank - (int)samples_b.size() : 0;
  int high = std::min((int)samples_a.size(), crank);
  while (low < high) {
    int i = low + (high - low) / 2;
    int j = crank - i;

    bool less = false;
    if (j > 0 && i < (int)samples_a.size()) {
      less = samples_a[i].value < samples_b[j - 1].value;
    } else if (j == 0) {
      less = false;
    }

    if (less) {
      low = i + 1;
    } else {
      high = i;
    }
  }
  return {low, crank - low};
}

std::pair<int, int> fine_co_rank(const std::vector<int> &a,
                                 const std::vector<int> &b, int lrank) {
  int low = (lrank > (int)b.size()) ? lrank - (int)b.size() : 0;
  int high = std::min((int)a.size(), lrank);
  while (low < high) {
    int i = low + (high - low) / 2;
    int j = lrank - i;
    bool less = false;
    if (j > 0 && i < (int)a.size()) {
      less = a[i] < b[j - 1];
    } else if (j == 0) {
      less = false;
    }
    if (less) {
      low = i + 1;
    } else {
      high = i;
    }
  }
  return {low, lrank - low};
}

std::vector<int> fetch_exact_data(const std::vector<int> &local_data,
                                  const std::vector<WideSample> &gather_s,
                                  int start_idx, int end_idx, int p, int step) {
  std::vector<int> req_start(p, 0);
  std::vector<int> req_count(p, 0);
  for (int i = start_idx; i < end_idx; ++i) {
    int chunk_idx = i / step;
    int offset = i % step;
    int node = gather_s[chunk_idx].node_rank;
    if (req_count[node] == 0) {
      req_start[node] = gather_s[chunk_idx].index + offset;
    }
    req_count[node]++;
  }
  std::vector<int> send_req(2 * p, 0);
  for (int i = 0; i < p; ++i) {
    send_req[2 * i] = req_start[i];
    send_req[2 * i + 1] = req_count[i];
  }

  std::vector<int> recv_req(2 * p, 0);
  MPI_Alltoall(send_req.data(), 2, MPI_INT, recv_req.data(), 2, MPI_INT,
               MPI_COMM_WORLD);

  std::vector<int> send_counts(p, 0), send_displs(p, 0);
  std::vector<int> recv_counts(p, 0), recv_displs(p, 0);

  for (int i = 0; i < p; ++i) {
    send_displs[i] = recv_req[2 * i];
    send_counts[i] = recv_req[2 * i + 1];
    recv_counts[i] = req_count[i];
  }

  recv_displs[0] = 0;
  for (int i = 1; i < p; ++i) {
    recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
  }

  int total_recv = end_idx - start_idx;
  std::vector<int> fetched_data(total_recv);
  MPI_Alltoallv(local_data.data(), send_counts.data(), send_displs.data(),
                MPI_INT, fetched_data.data(), recv_counts.data(),
                recv_displs.data(), MPI_INT, MPI_COMM_WORLD);

  return fetched_data;
}

std::vector<int> parallel_merge(const std::vector<int> &local_a,
                                const std::vector<int> &local_b, int node_rank,
                                int p, int res_size) {
  int n_local = local_a.size();
  int sample_size = n_local / p;
  // int global_n = p * n_local;

  std::vector<WideSample> sample_a(p);
  std::vector<WideSample> sample_b(p);

  for (int i = 0; i < p; i++) {
    sample_a[i] = WideSample(local_a[i * sample_size], i * sample_size, node_rank);
    sample_b[i] = WideSample(local_b[i * sample_size], i * sample_size, node_rank);
  }

  std::vector<WideSample> gather_s1(p * p);
  std::vector<WideSample> gather_s2(p * p);

  MPI_Allgather(sample_a.data(), 3 * p, MPI_INT, gather_s1.data(), 3 * p,
                MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(sample_b.data(), 3 * p, MPI_INT, gather_s2.data(), 3 * p,
                MPI_INT, MPI_COMM_WORLD);

  int grank_bg = node_rank * res_size;
  int grank_en = (node_rank + 1) * res_size;

  auto [cr_bg1, cr_bg2] = coarse_co_rank(gather_s1, gather_s2, grank_bg, sample_size);
  auto [cr_en1, cr_en2] = coarse_co_rank(gather_s1, gather_s2, grank_en, sample_size);
  
  // Expand search area slightly up to [cr - 1, cr + 1] establishing a tight
  // mathematically valid bound window
  int window_bg_start_a = std::max(0, (cr_bg1 - 1) * sample_size);
  int window_bg_end_a = std::min(p * p * sample_size, (cr_bg1 + 1) * sample_size);
  int window_bg_start_b = std::max(0, (cr_bg2 - 1) * sample_size);
  int window_bg_end_b = std::min(p * p * sample_size, (cr_bg2 + 1) * sample_size);

  std::vector<int> bg_window_a = fetch_exact_data(
      local_a, gather_s1, window_bg_start_a, window_bg_end_a, p, sample_size);
  std::vector<int> bg_window_b = fetch_exact_data(
      local_b, gather_s2, window_bg_start_b, window_bg_end_b, p, sample_size);

  auto[fr_bg1, fr_bg2] = fine_co_rank(bg_window_a, bg_window_b, grank_bg - (window_bg_start_a + window_bg_start_b));
  int exact_bg_a = window_bg_start_a + fr_bg1;
  int exact_bg_b = window_bg_start_b + fr_bg2;

  int window_en_start_a = std::max(0, (cr_en1 - 1) * sample_size);
  int window_en_end_a   = std::min(p * p * sample_size, (cr_en1 + 1) * sample_size);
  int window_en_start_b = std::max(0, (cr_en2 - 1) * sample_size);
  int window_en_end_b   = std::min(p * p * sample_size, (cr_en2 + 1) * sample_size);

  // redundant data fetch however it's capped to 3 * sample_size * 4 (each boundary) 
  std::vector<int> en_window_a = fetch_exact_data(local_a, gather_s1, window_en_start_a, window_en_end_a, p, sample_size);
  std::vector<int> en_window_b = fetch_exact_data(local_b, gather_s2, window_en_start_b, window_en_end_b, p, sample_size);
  auto [fr_en1, fr_en2] = fine_co_rank(en_window_a, en_window_b, grank_en - (window_en_start_a + window_en_start_b));
  int exact_en_a = window_en_start_a + fr_en1;
  int exact_en_b = window_en_start_b + fr_en2;


  std::vector<int> data_a = fetch_exact_data(local_a, gather_s1, exact_bg_a, exact_en_a, p, sample_size);
  std::vector<int> data_b = fetch_exact_data(local_b, gather_s2, exact_bg_b, exact_en_b, p, sample_size);

  std::vector<int> res(data_a.size() + data_b.size());
  std::merge(data_a.begin(), data_a.end(), data_b.begin(), data_b.end(),
             res.begin());

  std::cout << "Node: " << node_rank << " has " << res.size() << " elements."
            << std::endl;

  return res;
}

int main() {
  MPI_Init(nullptr, nullptr);

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
    std::binomial_distribution<> dist(2, 0.8);

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

    ///////////////////////////////////////////////////////////////////////////////////////////
    auto res = parallel_merge(slice_1, slice_2, world_rank, world_size, 2 * n);
    std::vector<int> sizes(world_size);
    int res_size = res.size();
    MPI_Gather(&res_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);

    std::vector<int> displ(world_size, 0);
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
    MPI_Scatter(nullptr, n, MPI_INT, slice_1.data(), n, MPI_INT, 0,
                MPI_COMM_WORLD);
    MPI_Scatter(nullptr, n, MPI_INT, slice_2.data(), n, MPI_INT, 0,
                MPI_COMM_WORLD);

    auto res = parallel_merge(slice_1, slice_2, world_rank, world_size, 2 * n);

    int res_size = res.size();
    MPI_Gather(&res_size, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(res.data(), res_size, MPI_INT, nullptr, nullptr, nullptr,
                MPI_INT, 0, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}