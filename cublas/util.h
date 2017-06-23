#pragma once

#include <sys/time.h>
#include <cstdint>
#include <vector>


inline uint64_t get_time(){
  struct timeval t;
  gettimeofday(&t, NULL);

  return static_cast<uint64_t>(t.tv_sec * 1000000 + t.tv_usec);
}

inline std::string shape_to_str(std::vector<int>& shape){
  std::string ret = "(";
  for(int &i : shape) { 
    ret += " " + std::to_string(i);
  }
  ret += ")";
  return ret;
}

inline std::string VLOG_vector(std::vector<int>& vec){ 
  std::string ret;
  for(auto & x : vec) { 
    ret += std::to_string(x) + ", " ;
  }
  return ret;
}


