1. 가상환경 키는법
cd ~/capstone2026/yolo
source vest/bin/activate

2. 실행
sudo ~/capstone2026/yolo/vest/bin/python vest_final_logic_v2.py

3. 현재 환경
가상환경: vest
Python: 3.10
torch: 2.8.0
CUDA: 12.6 사용 가능
numpy: 1.26.4
opencv-python: 4.10.0

4. 실험방법
유튜브 걷기 영상 켜놓고 캠 앞에 갖다댐 별차이 없는듯함

2026/05/31(민기찬 듀오)

vest_final_logic_v5_stream.py에 스트리밍 포함 구현 완료.
 스트리밍 확인할려면 학교서버의 mediaMTX가 실행상태여야함(http://학교서버IP:8889/vest/).
 
 사용중인 포트
RTSP listener opened on :8554
HLS listener opened on :8888
WebRTC listener opened on :8889




-- vue 키는법
conda activate vest
nvm use 22
cd ~/capstone2026/AIvestforblind/frontend
npm run dev -- --host 0.0.0.0 --port 5174
(or 5173)

-- event log 
conda activate vest
nvm use 22
cd ~/capstone2026/AIvestforblind
python event_ws_server.py

-- mediaMTX
cd ~/capstone2026/mediaMTX
./mediamtx

-- qwen 키는법 (Jetson)
cd ~/capstone2026/llama.cpp

./build/bin/llama-server \
  -m ~/capstone2026/yolo/models/qwen-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -c 1024 \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 8080
  
  
-- main
cd ~/capstone2026/yolo
source vest/bin/activate
python vest_final_logic_v6_7_korean_scene_natural.py
