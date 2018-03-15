#!/bin/bash

for((i=2;i<=6;i++))
do
    python id.py ./img001/img001-$i.jpg ./img001/img001-$i/
done

