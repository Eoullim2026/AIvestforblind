<template>
  <div class="dark-dashboard">
    <header class="top-nav">
      <div class="logo-area">
        <div class="logo-icon">〰️</div>
        <div class="titles">
          <h1>스마트조끼 관리자화면</h1>
          <p>Jetson Orin Nano 모니터 (시연 시뮬레이션)</p>
        </div>
      </div>
      <div class="header-actions">
        <button class="btn-outline" @click="showSettingModal = true">
          <span class="icon">🔗</span> 연결 설정
        </button>
        <button 
          class="btn-primary" 
          :class="{ 'is-connecting': isConnecting, 'is-connected': isConnected }"
          @click="toggleConnection"
          :disabled="isConnecting"
        >
          <span v-if="isConnecting">⏳ 연결 중...</span>
          <span v-else-if="isConnected">🟢 연결됨 (해제)</span>
          <span v-else>📡 장치 연결</span>
        </button>
      </div>
    </header>

    <main class="main-layout">
      <section class="video-panel">
        <div class="panel-header">
          <div class="panel-title">📷 라이브 피드 (모의 시연)</div>
          <div class="view-toggles">
            <button class="toggle active">RGB 카메라</button>
            <button class="toggle">깊이 맵 (MiDaS)</button>
          </div>
        </div>
        
        <div class="video-placeholder">
          <div v-if="!isConnected" class="no-signal">
            <span class="icon">🚫📡</span>
            <p>'장치 연결'을 눌러 시연 영상을 불러오세요.</p>
          </div>
          
          <div v-else class="video-active">
            <img src="/image_0.png" alt="시연용 보행 스냅샷" class="live-image" />
            
            <transition name="fade">
              <div v-if="currentAlert" class="danger-alert" :style="{ backgroundColor: currentAlert.color }">
                ⚠️ {{ currentAlert.message }}
              </div>
            </transition>

            <div 
              v-for="box in activeBoxes" 
              :key="box.id"
              class="bounding-box"
              :style="{ 
                top: box.y, left: box.x, width: box.w, height: box.h, 
                borderColor: box.color, backgroundColor: box.color + '33' 
              }"
            >
              <span class="box-label" :style="{ backgroundColor: box.color }">
                {{ box.label }} {{ box.confidence }}
              </span>
            </div>
          </div>
        </div>
      </section>

      <section class="log-panel">
        <div class="panel-header">
          <div class="panel-title">⏱️ 실시간 감지 로그 <span class="badge">{{ logs.length }}</span></div>
        </div>
        
        <div class="log-list">
          <transition-group name="list">
            <div v-for="log in logs.slice().reverse()" :key="log.id" class="log-card" :style="{ borderColor: log.color }">
              <div class="log-top">
                <div class="name-dir">
                  <span class="obj-name" :style="{ color: log.color }">{{ log.name }}</span>
                  <span class="direction">{{ log.direction }}</span>
                </div>
                <span class="time">{{ log.time }}</span>
              </div>
              <div class="log-bottom">
                <span class="distance">거리: <strong :style="{ color: log.color }">{{ log.distance }}</strong></span>
                <button class="play-video-btn" @click="openLogVideo(log)">▶ 영상</button>
                <span class="confidence">신뢰도<br><strong>{{ log.confidence }}</strong></span>
              </div>
            </div>
          </transition-group>
        </div>
      </section>
    </main>

    <div v-if="showSettingModal" class="modal-overlay" @click.self="showSettingModal = false">
      <div class="modal-content">
        <div class="modal-header">
          <h3>🔗 연결 설정</h3>
          <button class="close-btn" @click="showSettingModal = false">✕</button>
        </div>
        <div class="modal-body">
          <div class="input-group"><label>WebSocket URL</label><input type="text" value="ws://localhost:8765" /></div>
        </div>
        <div class="modal-footer">
          <button class="btn-cancel" @click="showSettingModal = false">취소</button>
          <button class="btn-save" @click="showSettingModal = false">✓ 저장</button>
        </div>
      </div>
    </div>

    <div v-if="showLogVideoModal" class="modal-overlay" @click.self="showLogVideoModal = false">
      <div class="modal-content video-modal">
        <div class="modal-header">
          <h3>🎞️ [{{ selectedLog?.name }}] 감지 기록 (전후 3초 시뮬레이션)</h3>
          <button class="close-btn" @click="showLogVideoModal = false">✕</button>
        </div>
        <div class="modal-body" style="padding: 0; background: #000;">
          <img src="/image_0.png" class="live-image" style="height: 300px;" />
          <div style="color: white; text-align: center; padding: 10px;">(시연용 짧은 클립 재생)</div>
        </div>
      </div>
    </div>

  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { DetectionLog, BoundingBox, DangerAlert } from '@/types/device'

const isConnected = ref(false)
const isConnecting = ref(false)
const showSettingModal = ref(false)
const showLogVideoModal = ref(false)
const selectedLog = ref<DetectionLog | null>(null)

const logs = ref<DetectionLog[]>([])
const activeBoxes = ref<BoundingBox[]>([])
const currentAlert = ref<DangerAlert | null>(null)
let simulationTimers: ReturnType<typeof setTimeout>[] = []

// [시뮬레이션 데이터]
const scenario = [
  { 
    time: 1000, 
    box: { id: 'b1', label: '가로등', confidence: '98%', x: '10%', y: '15%', w: '12%', h: '70%', color: '#ffea00' },
    log: { id: 1, name: '가로등', direction: '좌측 근접', distance: '1.5m', color: '#ffea00' },
    alert: { message: '좌측에 가로등 기둥 주의 (음성/진동)', color: '#f57f17' }
  },
  { 
    time: 3000, 
    box: { id: 'b2', label: '보행자', confidence: '92%', x: '45%', y: '25%', w: '20%', h: '50%', color: '#00e676' },
    log: { id: 2, name: '보행자', direction: '전방 중심', distance: '3.0m', color: '#00e676' },
    alert: null
  },
  { 
    time: 5500, 
    box: { id: 'b3', label: '자전거', confidence: '96%', x: '70%', y: '35%', w: '25%', h: '35%', color: '#ff1744' },
    log: { id: 3, name: '자전거', direction: '우측 접근', distance: '0.8m', color: '#ff1744' },
    alert: { message: '우측 자전거 매우 위험! 즉시 정지하세요! (음성+강력 진동)', color: '#b71c1c' }
  }
]

const toggleConnection = () => {
  if (isConnected.value) {
    isConnected.value = false
    logs.value = []
    activeBoxes.value = []
    currentAlert.value = null
    simulationTimers.forEach(t => clearTimeout(t))
    simulationTimers = []
  } else {
    isConnecting.value = true
    setTimeout(() => {
      isConnecting.value = false
      isConnected.value = true
      startSimulation()
    }, 1200)
  }
}

const startSimulation = () => {
  scenario.forEach(event => {
    const timerLog = setTimeout(() => {
      activeBoxes.value.push(event.box)
      const now = new Date()
      const timeStr = `${now.getHours()}:${now.getMinutes()}:${now.getSeconds()}`
      logs.value.push({ ...event.log, time: timeStr, confidence: event.box.confidence })

      if (event.alert) {
        currentAlert.value = event.alert
        const timerAlertClose = setTimeout(() => { currentAlert.value = null }, 3000)
        simulationTimers.push(timerAlertClose)
      }
    }, event.time)
    simulationTimers.push(timerLog)
  });
}

const openLogVideo = (log: DetectionLog) => {
  selectedLog.value = log
  showLogVideoModal.value = true
}
</script>

<style scoped>
.dark-dashboard { height: 100vh; display: flex; flex-direction: column; background-color: #121212; color: #e0e0e0; font-family: 'Pretendard', sans-serif; }
.top-nav { display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; background-color: #1a1a1a; border-bottom: 1px solid #333; z-index: 100; }
.logo-area { display: flex; align-items: center; gap: 12px; }
.logo-icon { font-size: 24px; color: #7e57c2; }
.titles h1 { font-size: 18px; margin: 0; font-weight: 700; color: #fff; }
.titles p { font-size: 11px; color: #888; margin: 2px 0 0 0; }
.header-actions { display: flex; gap: 12px; }
button { cursor: pointer; border-radius: 6px; font-weight: bold; font-size: 13px; padding: 8px 16px; transition: all 0.2s ease; border: none; outline: none; }
.btn-outline { background: transparent; border: 1px solid #555; color: #ccc; display: flex; align-items: center; gap: 6px; }
.btn-outline:hover { background: #333; }
.btn-primary { background-color: #00695c; color: #fff; width: 130px; text-align: center; }
.btn-primary:hover { background-color: #004d40; }
.btn-primary.is-connecting { background-color: #f57c00; cursor: wait; opacity: 0.8; }
.btn-primary.is-connected { background-color: #b71c1c; }

.main-layout { display: flex; gap: 20px; padding: 20px; flex: 1; overflow: hidden; }
.video-panel { flex: 2; background-color: #1e1e1e; border-radius: 12px; display: flex; flex-direction: column; border: 1px solid #333; overflow: hidden;}
.log-panel { flex: 1; min-width: 330px; background-color: #1e1e1e; border-radius: 12px; display: flex; flex-direction: column; border: 1px solid #333; overflow: hidden;}
.panel-header { display: flex; justify-content: space-between; align-items: center; padding: 14px 18px; border-bottom: 1px solid #333; background-color: #1a1a1a;}
.panel-title { font-size: 15px; font-weight: bold; color: #fff; display: flex; align-items: center; gap: 8px; }
.badge { background-color: #333; color: #aaa; padding: 2px 8px; border-radius: 12px; font-size: 11px; }
.view-toggles { display: flex; background-color: #121212; border-radius: 8px; padding: 3px; }
.view-toggles .toggle { background: transparent; color: #888; padding: 5px 10px; font-size: 12px;}
.view-toggles .toggle.active { background-color: #00695c; color: #fff; border-radius: 6px; }

.video-placeholder { flex: 1; background-color: #000; display: flex; justify-content: center; align-items: center; position: relative; overflow: hidden; }
.no-signal { text-align: center; color: #555; }
.no-signal .icon { font-size: 48px; display: block; margin-bottom: 15px; }
.video-active { width: 100%; height: 100%; position: relative; }

.live-image { width: 100%; height: 100%; object-fit: cover; display: block; filter: brightness(0.8); }

.danger-alert {
  position: absolute; top: 20px; left: 50%; transform: translateX(-50%);
  color: white; padding: 10px 20px; border-radius: 25px; font-weight: bold; font-size: 14px;
  box-shadow: 0 4px 15px rgba(0,0,0,0.5); z-index: 10; text-align: center; white-space: nowrap;
}

.bounding-box { position: absolute; border: 3px solid; border-radius: 4px; transition: all 0.3s ease; }
.box-label {
  position: absolute; top: -23px; left: -3px; color: #000; padding: 2px 6px; font-size: 11px; font-weight: bold; border-radius: 4px 4px 0 0; white-space: nowrap;
}

.log-list { flex: 1; overflow-y: auto; padding: 15px; display: flex; flex-direction: column; gap: 10px; background-color: #141414;}
.log-card { background-color: #1e1e1e; border: 1px solid; border-radius: 8px; padding: 12px; border-left-width: 4px; transition: all 0.3s; }
.log-card:hover { background-color: #252525; transform: translateY(-2px); }
.log-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.name-dir { display: flex; align-items: center; gap: 6px; }
.obj-name { font-weight: bold; font-size: 14px; }
.direction { background-color: #333; color: #aaa; font-size: 10px; padding: 2px 5px; border-radius: 3px; }
.time { font-size: 11px; color: #666; }
.log-bottom { display: flex; justify-content: space-between; align-items: center; gap: 5px;}
.distance { color: #888; font-size: 12px; }
.distance strong { font-size: 15px; }
.confidence { text-align: right; color: #555; font-size: 10px; line-height: 1.2; }
.confidence strong { color: #aaa; font-size: 13px; }
.play-video-btn { background-color: #333; color: #fff; padding: 5px 10px; font-size: 11px; border-radius: 4px; border: 1px solid #444; }
.play-video-btn:hover { background-color: #444; }

.list-enter-active { transition: all 0.5s ease-out; }
.list-enter-from { opacity: 0; transform: translateY(-20px); }
.fade-enter-active, .fade-leave-active { transition: opacity 0.4s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

.modal-overlay { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background-color: rgba(0, 0, 0, 0.7); display: flex; justify-content: center; align-items: center; z-index: 1000; backdrop-filter: blur(2px); }
.modal-content { background-color: #1e1e1e; width: 420px; border-radius: 12px; border: 1px solid #333; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
.video-modal { width: 550px; }
.modal-header { display: flex; justify-content: space-between; align-items: center; padding: 15px 20px; border-bottom: 1px solid #333; background-color: #1a1a1a;}
.modal-header h3 { margin: 0; font-size: 15px; color: #fff; font-weight: bold; }
.close-btn { background: transparent; color: #666; font-size: 18px; padding: 0; }
.modal-body { padding: 20px; }
.input-group label { display: block; font-size: 12px; color: #aaa; margin-bottom: 8px; }
.input-group input { width: 100%; box-sizing: border-box; background-color: #121212; border: 1px solid #333; color: #fff; padding: 10px 12px; border-radius: 6px; font-size: 13px; }
.modal-footer { display: flex; justify-content: flex-end; gap: 10px; padding: 15px 20px; border-top: 1px solid #333; background-color: #1a1a1a; }
.btn-cancel { background: transparent; color: #aaa; }
.btn-save { background-color: #00d2ff; color: #000; padding: 8px 20px; border-radius: 6px; font-weight: bold; }
</style>