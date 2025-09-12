#pragma once

#include <vector>
#include <chrono>
#include <string>

namespace streamkm {


    struct CoresetReducerChronoMetrics
    {   
        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = std::chrono::time_point<Clock>;

        struct MetricsSet {
            std::vector<CoresetReducerChronoMetrics> all_metrics;

            MetricsSet() = default;
            MetricsSet(CoresetReducerChronoMetrics &&m) : all_metrics(1) {
                all_metrics[0] = std::move(m);
            };
            
            void merge(MetricsSet &&other) {
                all_metrics.insert(all_metrics.end(),
                                   other.all_metrics.begin(),
                                   other.all_metrics.end());
            };

            void insert(CoresetReducerChronoMetrics &&m) {
                all_metrics.push_back(std::move(m));
            }

            template<typename PlotDuration = std::chrono::milliseconds>
            void print_avg() const {
                if (all_metrics.empty()) return;
                using Duration = std::chrono::nanoseconds;
                Duration total_node_pick = Duration::zero();
                Duration total_new_center = Duration::zero();
                Duration total_split = Duration::zero();
                Duration total_cost_update = Duration::zero();
                Duration total_init_time = Duration::zero();
                Duration total_final_coreset_time = Duration::zero();

                size_t total_iterations = 0;
                for (const auto &metrics : all_metrics) {
                    total_init_time += std::chrono::duration_cast<Duration>(metrics.end_init_time - metrics.start_init_time);
                    total_final_coreset_time += std::chrono::duration_cast<Duration>(metrics.end_final_coreset_time - metrics.start_final_coreset_time);
                    total_iterations += metrics.iterations.size();
                    for (const auto &it : metrics.iterations) {
                        total_node_pick += std::chrono::duration_cast<Duration>(it.end_node_pick_time - it.start_node_pick_time);
                        total_new_center += std::chrono::duration_cast<Duration>(it.end_new_center_time - it.start_new_center_time);
                        total_split += std::chrono::duration_cast<Duration>(it.end_split_time - it.start_split_time);
                        total_cost_update += std::chrono::duration_cast<Duration>(it.end_cost_update_time - it.start_cost_update_time);
                    }
                }   

                Duration total = total_node_pick + total_new_center + total_split + total_cost_update + total_init_time + total_final_coreset_time;

                auto to_plot = [](Duration d) {
                    return std::chrono::duration_cast<PlotDuration>(d).count();
                };

                // print TOTAL time
                std::cout << "Metrics over " << all_metrics.size() << " runs \n"
                    << "  Total init time: " <<  to_plot(total_init_time) << " (" << (100.0 * total_init_time.count() / total.count()) << "%)\n"
                    << "  Total final coreset time: " <<  to_plot(total_final_coreset_time) << " (" << (100.0 * total_final_coreset_time.count() / total.count()) << "%)\n"
                    << "     Total node pick time: " <<  to_plot(total_node_pick) << " (" << (100.0 * total_node_pick.count() / total.count()) << "%)\n"
                    << "     Total new center time: " <<  to_plot(total_new_center) << " (" << (100.0 * total_new_center.count() / total.count()) << "%)\n"
                    << "     Total split time: " <<  to_plot(total_split) << " (" << (100.0 * total_split.count() / total.count()) << "%)\n"
                    << "     Total cost update time: " <<  to_plot(total_cost_update) << " (" << (100.0 * total_cost_update.count() / total.count()) << "%)\n"
                    << "  Total time: " << to_plot(total) << "\n";
            }

            std::string toCsv() const {
                std::string csv = "run,iteration,node_pick_ns,new_center_ns,split_ns,cost_update_ns,node_size,inital_tree_cost,curr_tree_cost\n";
                for (size_t i = 0; i < all_metrics.size(); i++) {
                    const auto &metrics = all_metrics[i];
                    for (size_t j = 0; j < metrics.iterations.size(); j++) {
                        const auto &it = metrics.iterations[j];
                        auto node_pick_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(it.end_node_pick_time - it.start_node_pick_time).count();
                        auto new_center_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(it.end_new_center_time - it.start_new_center_time).count();
                        auto split_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(it.end_split_time - it.start_split_time).count();
                        auto cost_update_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(it.end_cost_update_time - it.start_cost_update_time).count();
                        csv += std::to_string(i) + "," 
                            + std::to_string(j)  + "," 
                            + std::to_string(node_pick_ns)  + "," 
                            + std::to_string(new_center_ns) + "," 
                            + std::to_string(split_ns) + "," 
                            + std::to_string(cost_update_ns) + ","
                            + std::to_string(it.node_size) + ","
                            + std::to_string(metrics.initial_tree_cost) + ","
                            + std::to_string(it.tree_cost)
                            + "\n";
                    }
                }
                return csv;
            }
        };

        struct IterMetrics
        {
            TimePoint start_node_pick_time,     end_node_pick_time;
            TimePoint start_new_center_time,    end_new_center_time;
            TimePoint start_split_time,         end_split_time;
            TimePoint start_cost_update_time,   end_cost_update_time;
            size_t    node_size = 0;
            double    tree_cost = 0.0;
        };
        
        std::vector<IterMetrics> iterations;

        IterMetrics curr_iter;
        TimePoint start_init_time,          end_init_time;
        TimePoint start_final_coreset_time, end_final_coreset_time;
        double    initial_tree_cost = 0.0;

        void start_init()           { start_init_time =                 Clock::now();  }
        void end_init()             { end_init_time =                   Clock::now();  }
        void start_final_coreset()  { start_final_coreset_time =        Clock::now();  }
        void end_final_coreset()    { end_final_coreset_time =          Clock::now();  }
        void start_node_pick()      { curr_iter.start_node_pick_time =  Clock::now(); }
        void end_node_pick()        { curr_iter.end_node_pick_time =    Clock::now();}
        void start_new_center()     { curr_iter.start_new_center_time = Clock::now(); }
        void end_new_center()       { curr_iter.end_new_center_time =   Clock::now(); }
        void start_split()          { curr_iter.start_split_time =      Clock::now(); }
        void end_split()            { curr_iter.end_split_time =        Clock::now(); }
        void start_cost_update()    { curr_iter.start_cost_update_time = Clock::now(); }
        void end_cost_update()      { curr_iter.end_cost_update_time =  Clock::now(); }
        void end_iteration()        { iterations.push_back(curr_iter);  }

        void set_initial_tree_cost(double c) { initial_tree_cost = c; }
        void set_tree_cost(double c) { curr_iter.tree_cost = c; }
        void set_node_size(size_t s) { curr_iter.node_size = s;  }
    };


    struct CoresetReducerNoMetrics
    {
        struct MetricsSet {
            MetricsSet() = default;
            MetricsSet(CoresetReducerNoMetrics &&) {}

            inline void merge(MetricsSet &&) const noexcept {};
            inline std::string toCsv() const noexcept{ return "run,iteration,node_pick_ns,new_center_ns,split_ns,cost_update_ns\n"; }

            template<typename PlotDuration = std::chrono::milliseconds>
            inline void print_avg() const noexcept {};

            inline void insert(CoresetReducerNoMetrics &&) const noexcept{};
        };

        inline void start_init() const noexcept           {}
        inline void end_init() const noexcept             {}
        inline void start_final_coreset() const noexcept  {}
        inline void end_final_coreset() const noexcept    {}
        inline void start_node_pick() const noexcept      {}
        inline void end_node_pick() const noexcept        {}
        inline void start_new_center() const noexcept     {}
        inline void end_new_center() const noexcept       {}
        inline void start_split() const noexcept          {}
        inline void end_split() const noexcept            {}
        inline void start_cost_update() const noexcept    {}
        inline void end_cost_update() const noexcept      {}
        inline void end_iteration() const noexcept        {}

        inline void set_initial_tree_cost(double) noexcept {}
        inline void set_tree_cost(double) noexcept        {}
        inline void set_node_size(size_t) noexcept        {}

    };



} // namespace streamkm