document.addEventListener('DOMContentLoaded', () => {
    const mapContainer = document.getElementById('map-container');
    const timeSlider = document.getElementById('time-slider');
    const timeDisplay = document.getElementById('time-display');
    const playButton = document.getElementById('play-button');
    const pauseButton = document.getElementById('pause-button');
    const timelineLabelsContainer = document.getElementById('timeline-labels');
    const flowSvg = document.getElementById('flow-svg');
    const svgDefs = flowSvg.querySelector('defs'); // 获取 <defs> 用于重绘

    // --- 1. 地点数据 (名称和百分比坐标) ---
    const locations = [
        { id: '507', name: "'507-人工智能联合创新实验室'", top: '15%', left: '25%' },
        { id: '506', name: "'506-人工智能联合创新实验室'", top: '10%', left: '40%' },
        { id: '505', name: "'505-人工智能联合创新实验室'", top: '8%', left: '55%' },
        { id: '504', name: "'504-人工智能联合创新实验室'", top: '10%', left: '70%' },
        { id: '503', name: "'503-无人驾驶实训室'", top: '15%', left: '80%' },
        { id: '502', name: "'502-XR联合创新实验室'", top: '30%', left: '88%' },
        { id: '501', name: "'501- 5G通信联合创新实验室'", top: '45%', left: '92%' },
        { id: 'west-hall', name: "'西侧展厅'", top: '58%', left: '89%' },
        { id: 'west-rest', name: "'西侧公共休息区'", top: '68%', left: '80%' },
        { id: 'washroom', name: "'洗手间走廊'", top: '70%', left: '65%' },
        { id: 'south-corridor', name: "'南侧走廊'", top: '78%', left: '50%' },
        { id: 'north-corridor', name: "'北侧走廊'", top: '75%', left: '35%' },
        { id: 'east-hall', name: "'东侧展厅'", top: '65%', left: '20%' },
        { id: 'east-rest', name: "'东侧公共休息区'", top: '55%', left: '12%' },
        { id: '510', name: "'510-中心机房'", top: '40%', left: '8%' },
        { id: '509', name: "'509-大数据/VR实训室'", top: '28%', left: '10%' }
    ];

    // 用于存储计算后的像素坐标
    const locationCoords = {};

    // --- 2. 创建地点标记和标签 ---
    locations.forEach(loc => {
        const point = document.createElement('div');
        point.className = 'location-point';
        point.style.top = loc.top;
        point.style.left = loc.left;
        point.dataset.id = loc.id;

        const label = document.createElement('span');
        label.className = 'location-label';
        label.textContent = loc.name;
        label.style.top = loc.top;
        label.style.left = loc.left;

        mapContainer.appendChild(point);
        mapContainer.appendChild(label);
    });

    // --- 3. 计算并存储像素坐标 ---
    function updateLocationCoordinates() {
        const mapRect = mapContainer.getBoundingClientRect();
        locations.forEach(loc => {
            // 找到对应的 DOM 元素来获取精确的计算后位置
            const pointElement = mapContainer.querySelector(`.location-point[data-id="${loc.id}"]`);
            if (pointElement) {
                // 计算相对于 mapContainer 的中心点坐标
                 const pointRect = pointElement.getBoundingClientRect();
                 const x = pointRect.left - mapRect.left + pointRect.width / 2;
                 const y = pointRect.top - mapRect.top + pointRect.height / 2;
                 locationCoords[loc.id] = { x, y };
            }
            // // 或者直接用百分比计算（可能不如上面精确）
            // const x = parseFloat(loc.left) / 100 * mapRect.width;
            // const y = parseFloat(loc.top) / 100 * mapRect.height;
            // locationCoords[loc.id] = { x, y };
        });
        // console.log("Updated Coords:", locationCoords); // Debug
    }

    // --- 4. 时间和滑块设置 ---
    const startTime = new Date('2023-09-04 00:00:00').getTime();
    const endTime = new Date('2023-09-19 16:00:00').getTime();
    const totalDuration = endTime - startTime;
    const sliderMax = parseInt(timeSlider.max, 10);

    // --- 5. 时间显示更新函数 (不变) ---
    function updateTimeDisplay(value) {
        const ratio = value / sliderMax;
        const currentTime = startTime + totalDuration * ratio;
        const date = new Date(currentTime);
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const hours = String(date.getHours()).padStart(2, '0');
        const minutes = String(date.getMinutes()).padStart(2, '0');
        timeDisplay.textContent = `Time: ${year}-${month}-${day} ${hours}:${minutes}`;
        // 返回当前时间戳用于预测
        return currentTime;
    }

    // --- 6. 时间轴标签生成函数 (不变) ---
    function generateTimelineLabels() {
        const numberOfLabels = 8;
        timelineLabelsContainer.innerHTML = '';
        for (let i = 0; i <= numberOfLabels; i++) {
            const ratio = i / numberOfLabels;
            const labelTime = startTime + totalDuration * ratio;
            const date = new Date(labelTime);
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            const hours = String(date.getHours()).padStart(2, '0');
            const labelSpan = document.createElement('span');
            labelSpan.textContent = `${date.getFullYear()}-${month}-${day} ${hours}:00`;
             if (i === 0) {
                 labelSpan.style.textAlign = 'left';
                 labelSpan.style.transform = 'translateX(0)';
                 labelSpan.style.left = '0%';
            } else if (i === numberOfLabels) {
                 labelSpan.style.textAlign = 'right';
                 labelSpan.style.transform = 'translateX(-100%)';
                 labelSpan.style.left = '100%';
            } else {
                 labelSpan.style.position = 'absolute';
                 labelSpan.style.left = `${ratio * 100}%`;
                 labelSpan.style.transform = 'translateX(-50%)';
            }
            timelineLabelsContainer.appendChild(labelSpan);
        }
    }

    // --- 7. 模拟预测流量 ---
    function getPredictedFlow(timestamp) {
        const date = new Date(timestamp);
        const hour = date.getHours();
        const dayOfWeek = date.getDay(); // 0 = Sunday, 6 = Saturday
        const flows = [];

        // 简化规则：
        // - 工作日 (1-5) 白天 (8-18) 有主要活动
        // - 周末和晚上活动减少

        const isWeekday = dayOfWeek >= 1 && dayOfWeek <= 5;
        const isWorkingHours = hour >= 8 && hour < 18;
        const isLunchTime = hour >= 12 && hour < 14;
        const isMorningRush = hour >= 8 && hour < 10;
        const isEveningRush = hour >= 16 && hour < 18;

        let baseVolume = isWeekday && isWorkingHours ? 5 : 1; // 基础流量

        // 模拟工作时间流向实验室/展厅
        if (isWeekday && isWorkingHours && !isLunchTime) {
            // 早上更多人流向各处
            const morningMultiplier = isMorningRush ? 1.5 : 1;
             // 走廊到实验室/展厅
            flows.push({ from: 'north-corridor', to: '507', volume: baseVolume * 0.5 * morningMultiplier });
            flows.push({ from: 'north-corridor', to: '509', volume: baseVolume * 0.5 * morningMultiplier });
            flows.push({ from: 'north-corridor', to: 'east-hall', volume: baseVolume * 0.8 * morningMultiplier });
            flows.push({ from: 'south-corridor', to: '503', volume: baseVolume * 0.5 * morningMultiplier });
            flows.push({ from: 'south-corridor', to: '501', volume: baseVolume * 0.5 * morningMultiplier });
            flows.push({ from: 'south-corridor', to: 'west-hall', volume: baseVolume * 0.8 * morningMultiplier });
             // 展厅到实验室
            flows.push({ from: 'east-hall', to: '509', volume: baseVolume * 0.3 });
            flows.push({ from: 'east-hall', to: '507', volume: baseVolume * 0.3 });
            flows.push({ from: 'west-hall', to: '501', volume: baseVolume * 0.3 });
            flows.push({ from: 'west-hall', to: '503', volume: baseVolume * 0.3 });
        }

        // 模拟午餐时间流向休息区
        if (isWeekday && isLunchTime) {
             // 从实验室/展厅去休息区
            flows.push({ from: 'east-hall', to: 'east-rest', volume: baseVolume * 1.5 });
            flows.push({ from: 'west-hall', to: 'west-rest', volume: baseVolume * 1.5 });
            flows.push({ from: '507', to: 'east-rest', volume: baseVolume * 0.5 });
            flows.push({ from: '503', to: 'west-rest', volume: baseVolume * 0.5 });
             // 休息区之间/去洗手间少量流动
            flows.push({ from: 'east-rest', to: 'washroom', volume: baseVolume * 0.2 });
            flows.push({ from: 'west-rest', to: 'washroom', volume: baseVolume * 0.3 });
        }

         // 模拟下班时间流出（简化为实验室流向走廊）
        if (isWeekday && isEveningRush) {
             flows.push({ from: '507', to: 'north-corridor', volume: baseVolume * 1.2 });
             flows.push({ from: '509', to: 'north-corridor', volume: baseVolume * 1.2 });
             flows.push({ from: '501', to: 'south-corridor', volume: baseVolume * 1.2 });
             flows.push({ from: '503', to: 'south-corridor', volume: baseVolume * 1.2 });
             flows.push({ from: 'east-hall', to: 'north-corridor', volume: baseVolume * 0.8 });
             flows.push({ from: 'west-hall', to: 'south-corridor', volume: baseVolume * 0.8 });
        }

        // 任何时间都可能去洗手间 (少量)
        flows.push({ from: 'north-corridor', to: 'washroom', volume: baseVolume * 0.1 });
        flows.push({ from: 'south-corridor', to: 'washroom', volume: baseVolume * 0.1 });


        // 随机加入少量背景流动
        const randomFlow = Math.random() * 0.5;
        flows.push({ from: 'east-hall', to: 'west-hall', volume: randomFlow });
        flows.push({ from: 'north-corridor', to: 'south-corridor', volume: randomFlow * 0.5 });


        // 返回非零流量
        return flows.filter(f => f.volume > 0.1); // 过滤掉非常小的流量
    }

    // --- 8. 更新流量可视化 ---
    function updateFlowVisualization(timestamp) {
        const predictedFlows = getPredictedFlow(timestamp);
        // console.log(`Time: ${new Date(timestamp)}, Flows: ${predictedFlows.length}`); // Debug

        // 清空旧线条 (保留 <defs>)
        flowSvg.innerHTML = '';
        flowSvg.appendChild(svgDefs.cloneNode(true)); // 把 <defs> 加回去

        if (Object.keys(locationCoords).length === 0) {
            // console.warn("Coordinates not ready yet."); // Debug
            return; // 如果坐标还没计算好，则不绘制
        }

        predictedFlows.forEach(flow => {
            const fromCoord = locationCoords[flow.from];
            const toCoord = locationCoords[flow.to];

            if (fromCoord && toCoord) {
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', fromCoord.x);
                line.setAttribute('y1', fromCoord.y);
                line.setAttribute('x2', toCoord.x);
                line.setAttribute('y2', toCoord.y);
                line.setAttribute('class', 'flow-line');

                // 根据流量调整粗细 (1 到 5 像素)
                const strokeWidth = Math.max(1, Math.min(1 + flow.volume * 0.8, 5));
                line.setAttribute('stroke-width', strokeWidth);

                // 添加箭头
                line.setAttribute('marker-end', 'url(#arrowhead)');

                flowSvg.appendChild(line);
            } else {
                // console.warn(`Coordinates missing for flow: ${flow.from} -> ${flow.to}`); // Debug
            }
        });
    }

    // --- 9. 滑块事件监听 ---
    timeSlider.addEventListener('input', () => {
        const currentTime = updateTimeDisplay(timeSlider.value);
        updateFlowVisualization(currentTime); // 更新流量可视化
    });

    // --- 10. 播放/暂停逻辑 ---
    let playInterval = null;
    let isPlaying = false;
    const playSpeed = 100; // ms per step (调慢一点，观察变化)
    const step = sliderMax / 1000; // 每次前进的步长 (调小一点，变化更平滑)

    function play() {
        if (isPlaying) return;
        isPlaying = true;
        playButton.disabled = true;
        pauseButton.disabled = false;

        if (parseInt(timeSlider.value, 10) >= sliderMax) {
            timeSlider.value = 0;
        }
        // 确保在开始播放前坐标已计算
        updateLocationCoordinates();

        playInterval = setInterval(() => {
            let currentValue = parseFloat(timeSlider.value); // 使用浮点数更平滑
            currentValue += step;

            if (currentValue >= sliderMax) {
                currentValue = sliderMax;
                pause();
            }

            timeSlider.value = currentValue;
            const currentTime = updateTimeDisplay(currentValue);
            updateFlowVisualization(currentTime); // 在播放中更新流量

        }, playSpeed);
    }

    function pause() {
        if (!isPlaying) return;
        isPlaying = false;
        clearInterval(playInterval);
        playButton.disabled = false;
        pauseButton.disabled = true;
    }

    playButton.addEventListener('click', play);
    pauseButton.addEventListener('click', pause);

    // --- 11. 窗口大小调整处理 ---
    window.addEventListener('resize', () => {
        updateLocationCoordinates(); // 重新计算坐标
        generateTimelineLabels(); // 重新生成时间标签
        const currentTime = updateTimeDisplay(timeSlider.value); // 获取当前时间
        updateFlowVisualization(currentTime); // 用新坐标重新绘制当前时间的流量
    });


    // --- 12. 初始化 ---
    updateLocationCoordinates(); // 首次计算坐标
    const initialTime = updateTimeDisplay(timeSlider.value);
    generateTimelineLabels();
    updateFlowVisualization(initialTime); // 初始绘制流量
    pauseButton.disabled = true;


});
