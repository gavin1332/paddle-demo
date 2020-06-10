#!/bin/bash

FETCH_VAR=$1

./demo --fetch_var=$FETCH_VAR 2>&1 | tee log.log
