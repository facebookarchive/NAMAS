export LUA_PATH="$LUA_PATH;?.lua"

th summary/run.lua \
 -modelFilename $2 \
 -inputf $1 \
 -length $3 \
 -blockRepeatWords 

