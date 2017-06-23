#!/bin/bash

nvprof -f --export-profile $1.nvvp ./a.out 
