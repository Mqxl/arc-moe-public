// Minimal backend probe: run with SYCL_GRAPH_FORCE_NATIVE_RECORDING=0 and 1.
// Compile: icpx -O2 -fsycl tools/probe-native-graph.cpp -o /tmp/arc-graph-probe
// No engine weights or graph-body operations are involved before begin_recording.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <iostream>
namespace graph_ext = sycl::ext::oneapi::experimental;
int main() {
 int fails=0, devices=0;
 for(auto device:sycl::device::get_devices(sycl::info::device_type::gpu)) {
  if(device.get_platform().get_backend()!=sycl::backend::ext_oneapi_level_zero)continue;
  ++devices;
  std::cout<<"device="<<device.get_info<sycl::info::device::name>()<<"\n";
  sycl::queue q(device,sycl::property::queue::in_order{});
  std::string stage="construct";
  int *p=sycl::malloc_device<int>(1,q);
  try {
   graph_ext::command_graph<graph_ext::graph_state::modifiable> graph(q.get_context(),device);
   stage="begin_empty_recording"; graph.begin_recording(q);
   stage="record_single_kernel"; q.single_task([=]{*p=63;});
   stage="end_recording"; graph.end_recording(q);
   stage="finalize"; auto executable=graph.finalize();
   stage="execute"; q.ext_oneapi_graph(executable).wait_and_throw();
   int value=0; q.memcpy(&value,p,sizeof(int)).wait_and_throw();
   std::cout<<"result="<<value<<"\n"; if(value!=63)++fails;
  } catch(const sycl::exception &e) {
   std::cout<<"stage="<<stage<<" code="<<e.code().value()<<" category="<<e.code().category().name()<<" error="<<e.what()<<"\n"; ++fails;
  }
  sycl::free(p,q);
 }
 return devices ? (fails?1:0) : 77;
}
