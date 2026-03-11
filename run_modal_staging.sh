#!/bin/bash
set -e

echo "Setting up clean Modal build context..."
rm -rf /root/modal_build_context
mkdir -p /root/modal_build_context/data

cd /root/cnn_lstm_v1
cp -r calibration eval inference jobs labels logging models selection strategy tests tuning utils config.yaml *.py /root/modal_build_context/
cp data/*.py /root/modal_build_context/data/

echo "Context size:"
du -sh /root/modal_build_context

cd /root/modal_build_context
source /root/cnn_lstm_v1/venv/bin/activate

TARGET_SCRIPT=${1:-modal_train.py}

echo "Starting Modal run: $TARGET_SCRIPT..."
modal run "$TARGET_SCRIPT"
