#!/bin/bash

export WINDOW=5
export OUT_DIR=$1/processed
export MDL_DIR=$1/models

export LUA_PATH="$LUA_PATH;$ABS/?.lua"

#bash $ABS/prep_torch_data.sh $2

mkdir -p $MDL_DIR

th -i $ABS/summary/train.lua -titleDir  $OUT_DIR/train/title/ \
 -articleDir  $OUT_DIR/train/article/ \
 -modelFilename  $MDL_DIR/$2 \
 -miniBatchSize  64 \
 -embeddingDim  64 \
 -bowDim  200 \
 -hiddenSize  64 \
 -epochs  20 \
 -learningRate 0.1 \
 -validArticleDir  $OUT_DIR/valid.filter/article/ \
 -validTitleDir  $OUT_DIR/valid.filter/title/ \
 -window  $WINDOW \
 -printEvery   100 \
 -encoderModel  "attenbow" \
 -attenPool  5 \
