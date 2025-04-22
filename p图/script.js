document.addEventListener('DOMContentLoaded', () => {
    const mapContainer = document.getElementById('map-container');
    const timeSlider = document.getElementById('time-slider');
    const timeDisplay = document.getElementById('time-display');
    const playButton = document.getElementById('play-button');
    const pauseButton = document.getElementById('pause-button');
    const timelineLabelsContainer = document.getElementById('timeline-labels');

    // 1. 定义地点数据 (名称和估算的百分比坐标 {top, left})
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

    // 2. 在地图上创建地点标记和标签
    locations.forEach(loc => {
        const point = document.createElement('div');
        point.className = 'location-point';
        point.style.top = loc.top;
        point.style.left = loc.left;
        point.dataset.id = loc.id; // 可以添加ID方便以后操作

        const label = document.createElement('span');
        label.className = 'location-label';
        label.textContent = loc.name;
        label.style.top = loc.top; // 标签也基于相同坐标定位，CSS会调整
        label.style.left = loc.left;

        mapContainer.appendChild(point);
        mapContainer.appendChild(label);
    });

    // 3. 时间和滑块设置
    const startTime = new Date('2023-09-04 00:00:00').getTime();
    const endTime = new Date('2023-09-19 16:00:00').getTime(); // 估算结束时间
    const totalDuration = endTime - startTime;
    const sliderMax = parseInt(timeSlider.max, 10); // 获取滑块的最大值

    // 4. 更新时间显示的函数
    function updateDisplay(value) {
        const ratio = value / sliderMax;
        const currentTime = startTime + totalDuration * ratio;
        const date = new Date(currentTime);

        // 格式化日期时间: YYYY-MM-DD HH:mm
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const hours = String(date.getHours()).padStart(2, '0');
        const minutes = String(date.getMinutes()).padStart(2, '0');

        timeDisplay.textContent = `Time: ${year}-${month}-${day} ${hours}:${minutes}`;
        timeSlider.value = value; // 确保滑块位置同步（特别是在播放时）
    }

    // 5. 生成时间轴标签
    function generateTimelineLabels() {
        const numberOfLabels = 8; // 你想要显示的主要时间标签数量
        timelineLabelsContainer.innerHTML = ''; // 清空旧标签

        for (let i = 0; i <= numberOfLabels; i++) {
            const ratio = i / numberOfLabels;
            const labelTime = startTime + totalDuration * ratio;
            const date = new Date(labelTime);
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            const hours = String(date.getHours()).padStart(2, '0');
            // const minutes = String(date.getMinutes()).padStart(2, '0');

            const labelSpan = document.createElement('span');
            // 显示更简洁的日期格式，例如 MM-DD HH
             // labelSpan.textContent = `${date.getFullYear()}-${month}-${day} ${hours}:${minutes}`;
             // 根据图片调整标签显示格式
            labelSpan.textContent = `${date.getFullYear()}-${month}-${day} ${hours}:00`;


            // 定位标签：第一个和最后一个靠边，中间的均匀分布
            if (i === 0) {
                 labelSpan.style.textAlign = 'left';
                 labelSpan.style.transform = 'translateX(0)'; // 重置transform
                 labelSpan.style.left = '0%';
            } else if (i === numberOfLabels) {
                 labelSpan.style.textAlign = 'right';
                 labelSpan.style.transform = 'translateX(-100%)'; // 重置transform
                 labelSpan.style.left = '100%';
            } else {
                 labelSpan.style.position = 'absolute';
                 labelSpan.style.left = `${ratio * 100}%`;
                 labelSpan.style.transform = 'translateX(-50%)';
            }
            // 微调第一个和最后一个标签使其与滑块边缘对齐
             if (i === 0) labelSpan.style.transform = 'translateX(0%)';
             if (i === numberOfLabels) labelSpan.style.transform = 'translateX(-100%)';


            timelineLabelsContainer.appendChild(labelSpan);
        }
    }


    // 6. 滑块事件监听
    timeSlider.addEventListener('input', () => {
        updateDisplay(timeSlider.value);
    });

    // 7. 播放/暂停逻辑
    let playInterval = null;
    let isPlaying = false;
    const playSpeed = 50; // ms per step, 越小越快
    const step = sliderMax / 500; // 每次前进的步长，可以调整

    function play() {
        if (isPlaying) return;
        isPlaying = true;
        playButton.disabled = true;
        pauseButton.disabled = false;

        // 如果滑块在末尾，则从头开始
        if (parseInt(timeSlider.value, 10) >= sliderMax) {
            timeSlider.value = 0;
        }

        playInterval = setInterval(() => {
            let currentValue = parseInt(timeSlider.value, 10);
            currentValue += step;

            if (currentValue >= sliderMax) {
                currentValue = sliderMax;
                pause(); // 到达末尾自动暂停
            }

            timeSlider.value = currentValue;
            updateDisplay(currentValue);

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

    // 8. 初始化
    updateDisplay(timeSlider.value); // 更新初始时间显示
    generateTimelineLabels(); // 生成时间轴标签
    pauseButton.disabled = true; // 初始时暂停按钮不可用

    // 当窗口大小改变时，重新生成时间轴标签以适应新的宽度
    window.addEventListener('resize', generateTimelineLabels);

});
