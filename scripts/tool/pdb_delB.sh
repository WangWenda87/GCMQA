#!/bin/bash

cat $1 | sed '/.\{16\}B.\{63\}/d' | sed -rn 's/(.{26})(.)(.*)/\1 \3/p'
