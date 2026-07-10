<template>
  <div class="dark-dashboard">
    <header class="top-nav">
      <div class="logo-area">
        <div class="logo-icon">〰️</div>
        <div class="titles">
          <h1>스마트조끼 관리자화면</h1>
          <p>Jetson Orin Nano 실시간 모니터</p>
        </div>
      </div>

      <div class="header-actions">
        <button class="btn-outline" @click="openSettingModal">
          <span class="icon">🔗</span> 연결 설정
        </button>

        <button
          class="btn-primary"
          :class="{ 'is-connecting': isConnecting, 'is-connected': isConnected }"
          @click="toggleConnection"
          :disabled="isConnecting"
        >
          <span v-if="isConnecting">⏳ 연결 중...</span>
          <span v-else-if="isConnected">🟢 연결됨</span>
          <span v-else>📡 스트림 연결</span>
        </button>
      </div>
    </header>

    <main class="main-layout">
      <section class="video-panel">
        <div class="panel-header">
          <div class="panel-title">📷 라이브 피드</div>

          <div class="stream-status" :class="{ active: isConnected }">
            <span class="status-dot"></span>
            <span>{{ isConnected ? 'LIVE' : 'OFFLINE' }}</span>
          </div>
        </div>

        <div class="video-placeholder">
          <div v-if="!isConnected" class="no-signal">
            <span class="icon">🚫📡</span>
            <p>'스트림 연결'을 눌러 Jetson 실시간 영상을 불러오세요.</p>
            <small>mediaMTX WebRTC: {{ activeStreamUrl }}</small>
          </div>

          <div v-else class="video-active">
            <iframe
              class="live-stream"
              :src="activeStreamUrl"
              allow="autoplay; fullscreen; picture-in-picture"
            ></iframe>

            <transition name="fade">
              <div
                v-if="currentAlert"
                class="danger-alert"
                :style="{ backgroundColor: currentAlert.color }"
              >
                ⚠️ {{ currentAlert.message }}
              </div>
            </transition>
          </div>
        </div>
      </section>

      <section class="log-panel">
        <div class="panel-header">
          <div class="panel-title">
            ⏱️ 실시간 감지 이벤트
            <span class="badge">{{ logs.length }}</span>
          </div>

          <div class="log-actions">
            <span class="ws-status" :class="wsStatus.toLowerCase()">
              {{ wsStatus }}
            </span>
            <button class="clear-log-btn" @click="clearLogs">지우기</button>
          </div>
        </div>

        <div class="log-list">
          <div v-if="logs.length === 0" class="empty-log">
            아직 감지 이벤트가 없습니다.
          </div>

          <transition-group name="list">
            <div
              v-for="log in logs.slice().reverse()"
              :key="log.id"
              class="log-card"
              :style="{ borderColor: log.color }"
            >
              <div class="log-top">
                <div class="name-dir">
                  <span class="obj-name" :style="{ color: log.color }">
                    {{ log.name }}
                  </span>

                  <span class="direction">{{ log.direction }}</span>

                  <span class="level-badge" :style="{ backgroundColor: log.color }">
                    {{ log.level }}
                  </span>

                  <span v-if="log.vibration" class="vibration-badge">
                    진동
                  </span>
                </div>

                <span class="time">{{ log.time }}</span>
              </div>

              <div class="log-bottom">
                <span class="distance">
                  거리:
                  <strong :style="{ color: log.color }">{{ log.distance }}</strong>
                </span>

                <button class="play-video-btn" @click="openLogVideo(log)">
                  ▶ 상세
                </button>

                <span class="confidence">
                  신뢰도<br />
                  <strong>{{ log.confidence }}</strong>
                </span>
              </div>

              <div v-if="log.message" class="log-message">
                {{ log.message }}
              </div>
            </div>
          </transition-group>
        </div>
      </section>
    </main>

    <div
      v-if="showSettingModal"
      class="modal-overlay"
      @click.self="cancelConnectionSettings"
    >
      <div class="modal-content">
        <div class="modal-header">
          <h3>🔗 연결 설정</h3>
          <button class="close-btn" @click="cancelConnectionSettings">✕</button>
        </div>

        <div class="modal-body">
          <div class="input-group">
            <label>Stream URL</label>
            <input type="text" v-model="draftStreamUrl" />
          </div>

          <div class="input-group" style="margin-top: 14px;">
            <label>Event WebSocket URL</label>
            <input type="text" v-model="draftWsUrl" />
          </div>

          <p class="setting-help">
            영상은 mediaMTX WebRTC 주소를 사용하고, 감지 이벤트 로그는 WebSocket으로 수신합니다.
          </p>
        </div>

        <div class="modal-footer">
          <button class="btn-cancel" @click="cancelConnectionSettings">취소</button>
          <button class="btn-save" @click="saveConnectionSettings">✓ 저장</button>
        </div>
      </div>
    </div>

    <div
      v-if="showLogVideoModal"
      class="modal-overlay"
      @click.self="showLogVideoModal = false"
    >
      <div class="modal-content video-modal">
        <div class="modal-header">
          <h3>🎞️ [{{ selectedLog?.name }}] 감지 이벤트 상세</h3>
          <button class="close-btn" @click="showLogVideoModal = false">✕</button>
        </div>

        <div class="modal-body log-detail-modal">
          <div class="detail-row">
            <span>객체</span>
            <strong>{{ selectedLog?.name }}</strong>
          </div>

          <div class="detail-row">
            <span>방향</span>
            <strong>{{ selectedLog?.direction }}</strong>
          </div>

          <div class="detail-row">
            <span>거리</span>
            <strong>{{ selectedLog?.distance }}</strong>
          </div>

          <div class="detail-row">
            <span>신뢰도</span>
            <strong>{{ selectedLog?.confidence }}</strong>
          </div>

          <div class="detail-row">
            <span>위험도</span>
            <strong>{{ selectedLog?.level }}</strong>
          </div>

          <div class="detail-row">
            <span>진동 발생</span>
            <strong>{{ selectedLog?.vibration ? '예' : '아니오' }}</strong>
          </div>

          <div class="detail-row">
            <span>감지 시각</span>
            <strong>{{ selectedLog?.time }}</strong>
          </div>

          <p class="setting-help">
            {{ selectedLog?.message || '추가 메시지가 없습니다.' }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const getDefaultStreamUrl = () => `${window.location.origin}/vest/`
const getDefaultWsUrl = () => {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'

  return `${wsProtocol}//${window.location.host}/ws`
}

const defaultStreamUrl = getDefaultStreamUrl()
const defaultWsUrl = getDefaultWsUrl()

const activeStreamUrl = ref(defaultStreamUrl)
const draftStreamUrl = ref(defaultStreamUrl)

const activeWsUrl = ref(defaultWsUrl)
const draftWsUrl = ref(defaultWsUrl)

const isConnected = ref(false)
const isConnecting = ref(false)
const wsStatus = ref('DISCONNECTED')

const showSettingModal = ref(false)
const showLogVideoModal = ref(false)
const selectedLog = ref(null)

const logs = ref([])
const currentAlert = ref(null)

let ws = null
let alertTimer = null
let reconnectTimer = null
let logId = 1

const openSettingModal = () => {
  draftStreamUrl.value = activeStreamUrl.value
  draftWsUrl.value = activeWsUrl.value
  showSettingModal.value = true
}

const saveConnectionSettings = () => {
  activeStreamUrl.value = normalizeStreamUrl(draftStreamUrl.value)
  activeWsUrl.value = normalizeWsUrl(draftWsUrl.value)
  showSettingModal.value = false

  if (isConnected.value) {
    disconnectStream()

    setTimeout(() => {
      connectStream()
    }, 300)
  }
}

const cancelConnectionSettings = () => {
  draftStreamUrl.value = activeStreamUrl.value
  draftWsUrl.value = activeWsUrl.value
  showSettingModal.value = false
}

const normalizeStreamUrl = (url) => {
  const trimmed = url.trim()

  if (!trimmed) {
    return defaultStreamUrl
  }

  if (trimmed.endsWith('/vest')) {
    return `${trimmed}/`
  }

  return trimmed
}

const normalizeWsUrl = (url) => {
  const trimmed = url.trim()

  if (!trimmed) {
    return defaultWsUrl
  }

  if (trimmed.startsWith('/')) {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'

    return `${wsProtocol}//${window.location.host}${trimmed}`
  }

  return trimmed
}

const toggleConnection = () => {
  if (isConnected.value) {
    disconnectStream()
    return
  }

  connectStream()
}

const connectStream = () => {
  isConnecting.value = true

  setTimeout(() => {
    isConnecting.value = false
    isConnected.value = true
    connectEventSocket()
  }, 500)
}

const disconnectStream = () => {
  isConnected.value = false
  wsStatus.value = 'DISCONNECTED'
  currentAlert.value = null

  if (alertTimer) {
    clearTimeout(alertTimer)
    alertTimer = null
  }

  if (reconnectTimer) {
    clearTimeout(reconnectTimer)
    reconnectTimer = null
  }

  if (ws) {
    ws.onopen = null
    ws.onmessage = null
    ws.onerror = null
    ws.onclose = null
    ws.close()
    ws = null
  }
}

const connectEventSocket = () => {
  if (!activeWsUrl.value) {
    wsStatus.value = 'NO URL'
    return
  }

  wsStatus.value = 'CONNECTING'

  try {
    ws = new WebSocket(activeWsUrl.value)

    ws.onopen = () => {
      wsStatus.value = 'CONNECTED'
      addSystemLog('이벤트 서버 연결됨', '#00e676')
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        addDetectionLog(data)
      } catch (error) {
        addSystemLog('이벤트 파싱 실패', '#ff9800')
        console.error('WebSocket message parse error:', error, event.data)
      }
    }

    ws.onerror = () => {
      wsStatus.value = 'ERROR'
      addSystemLog('이벤트 서버 오류', '#ff9800')
    }

    ws.onclose = () => {
      if (!isConnected.value) {
        return
      }

      wsStatus.value = 'DISCONNECTED'
      addSystemLog('이벤트 서버 연결 끊김', '#ff9800')

      reconnectTimer = setTimeout(() => {
        if (isConnected.value) {
          connectEventSocket()
        }
      }, 3000)
    }
  } catch (error) {
    wsStatus.value = 'ERROR'
    addSystemLog('WebSocket 연결 실패', '#ff1744')
    console.error('WebSocket connection error:', error)
  }
}

const addDetectionLog = (data) => {
  const level = normalizeLevel(data.level)
  const color = getLevelColor(level)

  const log = {
    id: logId++,
    name: data.name || data.object || '알 수 없음',
    direction: data.direction || '방향 미확인',
    distance: data.distance || '-',
    confidence: data.confidence || '-',
    level,
    message: data.message || '',
    vibration: Boolean(data.vibration),
    color,
    time: data.timestamp || getCurrentTime(),
  }

  logs.value.push(log)
  trimLogs()

  if (level === 'HIGH' || log.vibration) {
    showAlert(log.message || `${log.direction} ${log.name} 위험`)
  }
}

const addSystemLog = (message, color = '#777') => {
  logs.value.push({
    id: logId++,
    name: 'SYSTEM',
    direction: message,
    distance: '-',
    confidence: '-',
    level: 'INFO',
    message,
    vibration: false,
    color,
    time: getCurrentTime(),
  })

  trimLogs()
}

const trimLogs = () => {
  if (logs.value.length > 80) {
    logs.value.shift()
  }
}

const showAlert = (message) => {
  currentAlert.value = {
    message,
    color: '#b71c1c',
  }

  if (alertTimer) {
    clearTimeout(alertTimer)
  }

  alertTimer = setTimeout(() => {
    currentAlert.value = null
  }, 3000)
}

const normalizeLevel = (level) => {
  const value = String(level || 'INFO').toUpperCase()

  if (['HIGH', 'MEDIUM', 'LOW', 'INFO', 'UNKNOWN'].includes(value)) {
    return value
  }

  return 'INFO'
}

const getLevelColor = (level) => {
  switch (level) {
    case 'HIGH':
      return '#ff1744'
    case 'MEDIUM':
      return '#ffea00'
    case 'LOW':
      return '#00e676'
    case 'UNKNOWN':
      return '#ff9800'
    default:
      return '#777'
  }
}

const getCurrentTime = () => {
  const now = new Date()

  return [
    String(now.getHours()).padStart(2, '0'),
    String(now.getMinutes()).padStart(2, '0'),
    String(now.getSeconds()).padStart(2, '0'),
  ].join(':')
}

const clearLogs = () => {
  logs.value = []
}

const openLogVideo = (log) => {
  selectedLog.value = log
  showLogVideoModal.value = true
}
</script>

<style scoped>
.dark-dashboard {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #121212;
  color: #e0e0e0;
  font-family: 'Pretendard', sans-serif;
}

.top-nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  background-color: #1a1a1a;
  border-bottom: 1px solid #333;
  z-index: 100;
}

.logo-area {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-icon {
  font-size: 24px;
  color: #7e57c2;
}

.titles h1 {
  font-size: 18px;
  margin: 0;
  font-weight: 700;
  color: #fff;
}

.titles p {
  font-size: 11px;
  color: #888;
  margin: 2px 0 0 0;
}

.header-actions {
  display: flex;
  gap: 12px;
}

button {
  cursor: pointer;
  border-radius: 6px;
  font-weight: bold;
  font-size: 13px;
  padding: 8px 16px;
  transition: all 0.2s ease;
  border: none;
  outline: none;
}

.btn-outline {
  background: transparent;
  border: 1px solid #555;
  color: #ccc;
  display: flex;
  align-items: center;
  gap: 6px;
}

.btn-outline:hover {
  background: #333;
}

.btn-primary {
  background-color: #00695c;
  color: #fff;
  min-width: 135px;
  text-align: center;
}

.btn-primary:hover {
  background-color: #004d40;
}

.btn-primary.is-connecting {
  background-color: #f57c00;
  cursor: wait;
  opacity: 0.8;
}

.btn-primary.is-connected {
  background-color: #b71c1c;
}

.main-layout {
  display: flex;
  gap: 20px;
  padding: 20px;
  flex: 1;
  overflow: hidden;
}

.video-panel {
  flex: 2;
  background-color: #1e1e1e;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  border: 1px solid #333;
  overflow: hidden;
}

.log-panel {
  flex: 1;
  min-width: 350px;
  background-color: #1e1e1e;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  border: 1px solid #333;
  overflow: hidden;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 18px;
  border-bottom: 1px solid #333;
  background-color: #1a1a1a;
}

.panel-title {
  font-size: 15px;
  font-weight: bold;
  color: #fff;
  display: flex;
  align-items: center;
  gap: 8px;
}

.badge {
  background-color: #333;
  color: #aaa;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 11px;
}

.stream-status {
  display: flex;
  align-items: center;
  gap: 7px;
  color: #777;
  font-size: 12px;
  font-weight: bold;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #555;
}

.stream-status.active {
  color: #00e676;
}

.stream-status.active .status-dot {
  background-color: #00e676;
  box-shadow: 0 0 10px rgba(0, 230, 118, 0.8);
}

.video-placeholder {
  flex: 1;
  background-color: #000;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
  overflow: hidden;
}

.no-signal {
  text-align: center;
  color: #555;
  padding: 20px;
}

.no-signal .icon {
  font-size: 48px;
  display: block;
  margin-bottom: 15px;
}

.no-signal p {
  margin: 0 0 8px 0;
}

.no-signal small {
  color: #444;
  font-size: 11px;
}

.video-active {
  width: 100%;
  height: 100%;
  position: relative;
}

.live-stream {
  width: 100%;
  height: 100%;
  border: none;
  display: block;
  background: #000;
}

.danger-alert {
  position: absolute;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  color: white;
  padding: 10px 20px;
  border-radius: 25px;
  font-weight: bold;
  font-size: 14px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
  z-index: 10;
  text-align: center;
  white-space: nowrap;
}

.log-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.ws-status {
  font-size: 10px;
  font-weight: bold;
  padding: 4px 7px;
  border-radius: 999px;
  background-color: #333;
  color: #aaa;
}

.ws-status.connected {
  background-color: rgba(0, 230, 118, 0.15);
  color: #00e676;
}

.ws-status.connecting {
  background-color: rgba(255, 234, 0, 0.15);
  color: #ffea00;
}

.ws-status.disconnected,
.ws-status.error {
  background-color: rgba(255, 23, 68, 0.15);
  color: #ff1744;
}

.clear-log-btn {
  background-color: #333;
  color: #aaa;
  padding: 4px 8px;
  font-size: 10px;
  border-radius: 4px;
}

.clear-log-btn:hover {
  background-color: #444;
  color: #fff;
}

.log-list {
  flex: 1;
  overflow-y: auto;
  padding: 15px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  background-color: #141414;
}

.empty-log {
  color: #666;
  font-size: 13px;
  text-align: center;
  padding: 30px 0;
}

.log-card {
  background-color: #1e1e1e;
  border: 1px solid;
  border-radius: 8px;
  padding: 12px;
  border-left-width: 4px;
  transition: all 0.3s;
}

.log-card:hover {
  background-color: #252525;
  transform: translateY(-2px);
}

.log-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  gap: 8px;
}

.name-dir {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.obj-name {
  font-weight: bold;
  font-size: 14px;
}

.direction {
  background-color: #333;
  color: #aaa;
  font-size: 10px;
  padding: 2px 5px;
  border-radius: 3px;
}

.level-badge {
  color: #000;
  font-size: 9px;
  font-weight: bold;
  padding: 2px 5px;
  border-radius: 4px;
}

.vibration-badge {
  background-color: #b71c1c;
  color: #fff;
  font-size: 9px;
  font-weight: bold;
  padding: 2px 5px;
  border-radius: 4px;
}

.time {
  font-size: 11px;
  color: #666;
  white-space: nowrap;
}

.log-bottom {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 5px;
}

.distance {
  color: #888;
  font-size: 12px;
}

.distance strong {
  font-size: 15px;
}

.confidence {
  text-align: right;
  color: #555;
  font-size: 10px;
  line-height: 1.2;
}

.confidence strong {
  color: #aaa;
  font-size: 13px;
}

.play-video-btn {
  background-color: #333;
  color: #fff;
  padding: 5px 10px;
  font-size: 11px;
  border-radius: 4px;
  border: 1px solid #444;
}

.play-video-btn:hover {
  background-color: #444;
}

.log-message {
  color: #aaa;
  font-size: 11px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #292929;
}

.list-enter-active {
  transition: all 0.5s ease-out;
}

.list-enter-from {
  opacity: 0;
  transform: translateY(-20px);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.4s;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background-color: rgba(0, 0, 0, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  backdrop-filter: blur(2px);
}

.modal-content {
  background-color: #1e1e1e;
  width: 420px;
  border-radius: 12px;
  border: 1px solid #333;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
}

.video-modal {
  width: 550px;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  border-bottom: 1px solid #333;
  background-color: #1a1a1a;
}

.modal-header h3 {
  margin: 0;
  font-size: 15px;
  color: #fff;
  font-weight: bold;
}

.close-btn {
  background: transparent;
  color: #666;
  font-size: 18px;
  padding: 0;
}

.modal-body {
  padding: 20px;
}

.input-group label {
  display: block;
  font-size: 12px;
  color: #aaa;
  margin-bottom: 8px;
}

.input-group input {
  width: 100%;
  box-sizing: border-box;
  background-color: #121212;
  border: 1px solid #333;
  color: #fff;
  padding: 10px 12px;
  border-radius: 6px;
  font-size: 13px;
}

.setting-help {
  color: #777;
  font-size: 12px;
  line-height: 1.5;
  margin: 12px 0 0 0;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 15px 20px;
  border-top: 1px solid #333;
  background-color: #1a1a1a;
}

.btn-cancel {
  background: transparent;
  color: #aaa;
}

.btn-save {
  background-color: #00d2ff;
  color: #000;
  padding: 8px 20px;
  border-radius: 6px;
  font-weight: bold;
}

.log-detail-modal {
  background: #141414;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid #292929;
  padding: 10px 0;
  font-size: 13px;
}

.detail-row span {
  color: #777;
}

.detail-row strong {
  color: #fff;
}

@media (max-width: 900px) {
  .main-layout {
    flex-direction: column;
    overflow-y: auto;
  }

  .log-panel {
    min-width: unset;
    min-height: 280px;
  }

  .video-panel {
    min-height: 420px;
  }

  .top-nav {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }

  .header-actions {
    width: 100%;
  }

  .btn-primary,
  .btn-outline {
    flex: 1;
    justify-content: center;
  }
}
</style>
