#! /bin/bash
for i in img/style*; do
    python src/style.py content_small.jpg $i
    python src/style.py content_small.jpg $i -r -a 50
done
