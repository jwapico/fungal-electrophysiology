import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const V_MIN = -70;
const V_MAX = 30;
const SPIKE_DURATION_MS = 20;
const SPIKE_DECAY_TAU = SPIKE_DURATION_MS / 3;
const CUBE_SIZE = 0.4;          // doubled from 0.2
const SPHERE_SIZE = 1.25;        // doubled from 1.0

function voltageToColor(voltage: number): THREE.Color {
    const t = Math.max(0, Math.min(1, (voltage - V_MIN) / (V_MAX - V_MIN)));
    const color = new THREE.Color();
    color.setHSL(0.667 * (1 - t), 1, 0.5);
    return color;
}

interface SimulationData {
    metadata: {
        duration: number;
        dt: number;
        num_neurons: number;
        num_steps: number;
        spike_threshold: number;
    };
    nodes: Array<{
        id: number;
        position: [number, number, number];
    }>;
    edges: Array<{
        source: number;
        target: number;
        weight: number;
        distance: number;
        section_id: number;
    }>;
    time: number[];
    voltages: number[][];
    spikes: { [neuron: string]: number[] };
}

async function loadData(): Promise<SimulationData> {
    const response = await fetch('simulation_data.json');
    if (!response.ok) {
        throw new Error(`Failed to load data: ${response.statusText}`);
    }
    return await response.json();
}

class Visualizer {
    private scene: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private renderer: THREE.WebGLRenderer;
    private controls: OrbitControls;
    private data: SimulationData | null = null;
    private edgeLines: THREE.Line[] = [];
    private validEdgeData: Array<{edge: any, line: THREE.Line, colorAttr: THREE.BufferAttribute}> = [];
    private nodeIdToSphereIndex: Map<number, number> = new Map();
    private edgeColorAttributes: THREE.BufferAttribute[] = [];
    private nodeCubes: THREE.Mesh[] = [];
    private nodeSpheres: THREE.Mesh[] = [];
    private timeIndex = 0;
    private playing = false;
    private animationId = 0;
    private speedMultiplier = 1.0;
    private currentTimeMs = 0;
    private spikeTimesMap: Map<number, number[]> = new Map();
    private yellow = new THREE.Color(1, 1, 0);
    private red = new THREE.Color(1, 0, 0);
    private rasterPlot: RasterPlot | null = null;

    constructor() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x111111);
        this.camera = new THREE.PerspectiveCamera(
            60,
            window.innerWidth / window.innerHeight,
            0.1,
            10000
        );
        this.camera.position.set(0, 0, 500);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(this.renderer.domElement);
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        window.addEventListener('resize', () => this.onResize());
        this.setupUI();
    }

    private setupUI() {
        const playPauseButton = document.getElementById('playPause') as HTMLButtonElement;
        const resetButton = document.getElementById('reset') as HTMLButtonElement;
        const timeSlider = document.getElementById('timeSlider') as HTMLInputElement;
        const timeDisplay = document.getElementById('timeDisplay')!;
        const speedSlider = document.getElementById('speedSlider') as HTMLInputElement;
        const speedDisplay = document.getElementById('speedDisplay')!;

        playPauseButton.addEventListener('click', () => {
            this.playing = !this.playing;
            playPauseButton.textContent = this.playing ? 'Pause' : 'Play';
            if (this.playing) this.animate();
        });

        resetButton.addEventListener('click', () => {
            this.currentTimeMs = 0;
            this.timeIndex = 0;
            timeSlider.value = '0';
            timeDisplay.textContent = '0';
            this.updateColors();
            if (this.rasterPlot)
                this.rasterPlot.update(this.currentTimeMs, this.playing);
        });

        timeSlider.addEventListener('input', (e) => {
            this.currentTimeMs = parseFloat((e.target as HTMLInputElement).value);
            this.timeIndex = Math.floor(this.currentTimeMs / this.data!.metadata.dt);
            timeDisplay.textContent = this.currentTimeMs.toFixed(1);
            this.updateColors();
            if (this.rasterPlot) {
                this.rasterPlot.update(this.currentTimeMs, this.playing);
            }
        });

        speedSlider.addEventListener('input', (e) => {
            this.speedMultiplier = parseFloat((e.target as HTMLInputElement).value);
            speedDisplay.textContent = this.speedMultiplier.toFixed(1);
        });
    }

    private onResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    async init() {
        this.data = await loadData();
        this.prepareSpikeData();
        const rasterContainer = document.getElementById('rasterContainer') as HTMLDivElement;
        this.rasterPlot = new RasterPlot(rasterContainer, this.data.spikes, this.data.metadata.duration);
        this.createGeometry();
        this.updateColors();
    }

    private prepareSpikeData() {
        if (!this.data) return;
        const spikes = this.data.spikes;
        for (const neuronStr in spikes) {
            const neuron = parseInt(neuronStr);
            const times = spikes[neuronStr].sort((a, b) => a - b);
            this.spikeTimesMap.set(neuron, times);
        }
    }

    private spikeIntensity(neuron: number, timeMs: number): number {
        const times = this.spikeTimesMap.get(neuron);
        if (!times || times.length === 0) return 0;
        let lastSpike = -Infinity;
        for (let i = times.length - 1; i >= 0; i--) {
            if (times[i] <= timeMs) {
                lastSpike = times[i];
                break;
            }
        }
        if (lastSpike === -Infinity) return 0;
        const delta = timeMs - lastSpike;
        if (delta > SPIKE_DURATION_MS) return 0;
        return Math.exp(-delta / SPIKE_DECAY_TAU);
    }

    private createGeometry() {
        if (!this.data) return;

        const { nodes, edges } = this.data;
        const nodeMap = new Map<number, THREE.Vector3>();

        nodes.forEach((node, index) => {
            const [x, y, z] = node.position;
            const position = new THREE.Vector3(x, y, z);
            nodeMap.set(node.id, position);
            this.nodeIdToSphereIndex.set(node.id, index);

            const cubeGeometry = new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE);
            const cubeMaterial = new THREE.MeshBasicMaterial({ color: this.red });
            const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);
            cube.position.copy(position);
            this.scene.add(cube);
            this.nodeCubes.push(cube);

            const sphereGeometry = new THREE.SphereGeometry(SPHERE_SIZE, 16, 16);
            const sphereMaterial = new THREE.MeshBasicMaterial({
                color: this.yellow,
                transparent: true,
                opacity: 0
            });
            const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
            sphere.position.copy(position);
            this.scene.add(sphere);
            this.nodeSpheres.push(sphere);
        });

        // Clear previous edge data
        this.validEdgeData = [];

        edges.forEach(edge => {
            const source = nodeMap.get(edge.source);
            const target = nodeMap.get(edge.target);
            if (!source || !target) return;

            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array([
                source.x, source.y, source.z,
                target.x, target.y, target.z
            ]);
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

            const colors = new Float32Array(6);
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

            const material = new THREE.LineBasicMaterial({
                vertexColors: true,
                linewidth: 1,
                transparent: true,
                opacity: 0.8
            });

            const line = new THREE.Line(geometry, material);
            this.scene.add(line);
            const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute;
            this.validEdgeData.push({ edge, line, colorAttr });
        });
    }

    private updateColors() {
        if (!this.data) return;

        const volt = this.data.voltages;
        const timeIdx = Math.min(this.timeIndex, this.data.time.length - 1);

        // Update edge colors
        this.validEdgeData.forEach(({ edge, colorAttr }) => {
            const sourceVolt = volt[edge.source][timeIdx];
            const targetVolt = volt[edge.target][timeIdx];
            let sourceColor = voltageToColor(sourceVolt);
            let targetColor = voltageToColor(targetVolt);

            const flashIntensity = Math.max(
                this.spikeIntensity(edge.source, this.currentTimeMs),
                this.spikeIntensity(edge.target, this.currentTimeMs)
            );
            if (flashIntensity > 0) {
                sourceColor.lerp(this.yellow, flashIntensity);
                targetColor.lerp(this.yellow, flashIntensity);
            }

            colorAttr.setXYZ(0, sourceColor.r, sourceColor.g, sourceColor.b);
            colorAttr.setXYZ(1, targetColor.r, targetColor.g, targetColor.b);
            colorAttr.needsUpdate = true;
        });

        // Update node sphere opacity
        for (let i = 0; i < this.nodeSpheres.length; i++) {
            const node = this.data.nodes[i];
            const intensity = this.spikeIntensity(node.id, this.currentTimeMs);
            this.nodeSpheres[i].material.opacity = intensity;
        }
    }

    private animate = () => {
        if (!this.playing) return;

        this.animationId = requestAnimationFrame(this.animate);

        if (this.data) {
            const dt = this.data.metadata.dt;
            const maxTime = this.data.time[this.data.time.length - 1];
            
            this.currentTimeMs += dt * this.speedMultiplier;
            if (this.currentTimeMs > maxTime) {
                this.currentTimeMs = maxTime;
                this.playing = false;
                document.getElementById('playPause')!.textContent = 'Play';
            }

            this.timeIndex = Math.floor(this.currentTimeMs / dt);
            const slider = document.getElementById('timeSlider') as HTMLInputElement;
            slider.value = this.currentTimeMs.toString();
            document.getElementById('timeDisplay')!.textContent = this.currentTimeMs.toFixed(1);

            this.updateColors();
            if (this.rasterPlot)
                this.rasterPlot.update(this.currentTimeMs, this.playing);
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    };

    start() {
        this.animate();
    }
}

class RasterPlot {
    private offscreen: HTMLCanvasElement;
    private offCtx: CanvasRenderingContext2D;
    private display: HTMLCanvasElement;
    private displayCtx: CanvasRenderingContext2D;
    private container: HTMLDivElement;
    private resizeHandle: HTMLDivElement;
    private scrollbar: HTMLDivElement;
    private scrollThumb: HTMLDivElement;
    private spikes: { [neuron: string]: number[] };
    private duration: number;
    private windowSize: number = 500; // ms
    private windowStart: number = 0;
    private isDragging: boolean = false;
    private dragStartX: number = 0;
    private dragStartWindowStart: number = 0;
    private minimized: boolean = false;
    private defaultHeight: number = 200;
    private spikingNeuronIds: number[] = [];
    private neuronToRow: Map<number, number> = new Map();
    private currentTimeMs: number = 0;
    
    // New properties for vertical scrolling and row height
    private rowHeight: number = 2; // pixels per neuron
    private scrollOffset: number = 0; // index of first visible neuron
    private totalRows: number = 0; // total number of spiking neurons
    private visibleNeurons: number = 0; // number of neurons visible at once
    private isScrolling: boolean = false;
    private scrollStartY: number = 0;
    private scrollStartOffset: number = 0;
    
    constructor(container: HTMLDivElement, spikes: { [neuron: string]: number[] }, duration: number) {
        this.container = container;
        this.spikes = spikes;
        this.duration = duration;
        this.display = document.getElementById('rasterCanvas') as HTMLCanvasElement;
        this.displayCtx = this.display.getContext('2d')!;
        this.resizeHandle = document.getElementById('rasterResize') as HTMLDivElement;
        this.scrollbar = document.getElementById('rasterScrollbar') as HTMLDivElement;
        this.scrollThumb = document.getElementById('rasterScrollThumb') as HTMLDivElement;
        
        // Filter and sort spiking neurons by ID
        this.spikingNeuronIds = Object.keys(spikes)
            .map(Number)
            .sort((a, b) => a - b);
        this.totalRows = this.spikingNeuronIds.length;
        
        // Build mapping from neuron ID to row index
        this.spikingNeuronIds.forEach((neuronId, index) => {
            this.neuronToRow.set(neuronId, index);
        });
        
        // Create offscreen canvas: 1 px per ms, 1 px per spiking neuron
        this.offscreen = document.createElement('canvas');
        this.offscreen.width = Math.ceil(duration); // total ms
        this.offscreen.height = this.totalRows || 1; // at least 1 row
        this.offCtx = this.offscreen.getContext('2d')!;
        
        this.drawAllSpikes();
        this.setupEvents();
        this.resize();
        this.windowStart = 0; // start at time 0
    }

    private drawAllSpikes() {
        this.offCtx.clearRect(0, 0, this.offscreen.width, this.offscreen.height);
        this.offCtx.fillStyle = 'white';
        
        for (const neuronId of this.spikingNeuronIds) {
            const times = this.spikes[neuronId.toString()];
            if (!times) continue;
            const row = this.neuronToRow.get(neuronId);
            if (row === undefined) continue;
            
            times.forEach(t => {
                const x = Math.floor(t);
                this.offCtx.fillRect(x, row, 1, 1);
            });
        }
    }

    private setupEvents() {
        // Drag to pan (horizontal)
        this.display.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.dragStartX = e.clientX;
            this.dragStartWindowStart = this.windowStart;
        });
        window.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const dx = e.clientX - this.dragStartX;
                const dt = dx * (this.windowSize / this.display.clientWidth);
                this.windowStart = Math.max(0, Math.min(this.duration - this.windowSize, this.dragStartWindowStart - dt));
                this.draw();
            } else if (this.isScrolling) {
                const dy = e.clientY - this.scrollStartY;
                const rowsMoved = dy / this.rowHeight;
                this.scrollOffset = Math.max(0, Math.min(this.totalRows - this.visibleNeurons, this.scrollStartOffset + rowsMoved));
                this.updateScrollbar();
                this.draw();
            }
        });
        window.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.isScrolling = false;
        });

        // Mouse wheel for row height adjustment and vertical scrolling
        this.display.addEventListener('wheel', (e) => {
            e.preventDefault();
            if (e.ctrlKey || e.metaKey) {
                // Adjust row height
                const zoomFactor = 1.1;
                if (e.deltaY < 0) {
                    this.rowHeight = Math.min(10, this.rowHeight * zoomFactor);
                } else {
                    this.rowHeight = Math.max(1, this.rowHeight / zoomFactor);
                }
            } else {
                // Vertical scroll
                const scrollAmount = e.deltaY / this.rowHeight;
                this.scrollOffset = Math.max(0, Math.min(this.totalRows - this.visibleNeurons, this.scrollOffset + scrollAmount));
            }
            this.updateScrollbar();
            this.draw();
        });

        // Scrollbar thumb dragging
        this.scrollThumb.addEventListener('mousedown', (e) => {
            e.preventDefault();
            this.isScrolling = true;
            this.scrollStartY = e.clientY;
            this.scrollStartOffset = this.scrollOffset;
        });

        // Click on scrollbar track to jump
        this.scrollbar.addEventListener('mousedown', (e) => {
            if (e.target === this.scrollThumb) return;
            const rect = this.scrollbar.getBoundingClientRect();
            const clickY = e.clientY - rect.top;
            const thumbHeight = (this.visibleNeurons / this.totalRows) * this.scrollbar.clientHeight;
            const trackHeight = this.scrollbar.clientHeight - thumbHeight;
            const fraction = clickY / trackHeight;
            this.scrollOffset = Math.floor(fraction * (this.totalRows - this.visibleNeurons));
            this.updateScrollbar();
            this.draw();
        });

        // Resize handle
        this.resizeHandle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            const startY = e.clientY;
            const startHeight = this.container.offsetHeight;
            const onMove = (moveEvent: MouseEvent) => {
                const dy = startY - moveEvent.clientY;
                const newHeight = Math.max(50, Math.min(window.innerHeight * 0.5, startHeight + dy));
                this.container.style.height = newHeight + 'px';
                this.resize();
            };
            const onUp = () => {
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('mouseup', onUp);
            };
            window.addEventListener('mousemove', onMove);
            window.addEventListener('mouseup', onUp);
        });

        // Double‑click to toggle minimized
        this.resizeHandle.addEventListener('dblclick', () => {
            this.minimized = !this.minimized;
            if (this.minimized) {
                this.container.style.height = '4px';
            } else {
                this.container.style.height = this.defaultHeight + 'px';
            }
            this.resize();
        });
    }

    private resize() {
        this.display.width = this.display.clientWidth;
        this.display.height = this.display.clientHeight - 4; // account for resize handle
        this.visibleNeurons = Math.floor(this.display.height / this.rowHeight);
        if (this.visibleNeurons > this.totalRows) {
            this.visibleNeurons = this.totalRows;
            this.scrollOffset = 0;
        } else {
            // Ensure scrollOffset is within bounds
            this.scrollOffset = Math.max(0, Math.min(this.totalRows - this.visibleNeurons, this.scrollOffset));
        }
        this.updateScrollbar();
        this.draw();
    }

    private updateScrollbar() {
        if (this.totalRows <= this.visibleNeurons) {
            this.scrollbar.style.display = 'none';
            return;
        }
        this.scrollbar.style.display = 'block';
        const thumbHeight = (this.visibleNeurons / this.totalRows) * this.scrollbar.clientHeight;
        const maxScroll = this.totalRows - this.visibleNeurons;
        const thumbTop = maxScroll > 0 ? (this.scrollOffset / maxScroll) * (this.scrollbar.clientHeight - thumbHeight) : 0;
        this.scrollThumb.style.height = thumbHeight + 'px';
        this.scrollThumb.style.top = thumbTop + 'px';
    }

    public update(currentTimeMs: number, isPlaying: boolean) {
        this.currentTimeMs = currentTimeMs;
        if (isPlaying && !this.isDragging) {
            // Auto‑scroll: center window on current time
            this.windowStart = Math.max(0, Math.min(this.duration - this.windowSize, currentTimeMs - this.windowSize / 2));
        }
        this.draw();
    }

    private draw() {
        const { width, height } = this.display;
        this.displayCtx.clearRect(0, 0, width, height);
        
        // Calculate source rectangle
        const srcX = Math.max(0, Math.min(this.offscreen.width - this.windowSize, Math.floor(this.windowStart)));
        const srcWidth = Math.min(this.windowSize, this.offscreen.width - srcX);
        const srcY = this.scrollOffset;
        const srcHeight = this.visibleNeurons;
        
        // Destination rectangle
        const destHeight = srcHeight * this.rowHeight;
        
        // Draw the raster portion
        this.displayCtx.imageSmoothingEnabled = false;
        this.displayCtx.drawImage(
            this.offscreen,
            srcX, srcY, srcWidth, srcHeight,
            0, 0, width, destHeight
        );
        
        // Draw yellow time indicator line
        const currentTimeX = (this.currentTimeMs - this.windowStart) * (width / this.windowSize);
        if (currentTimeX >= 0 && currentTimeX <= width) {
            this.displayCtx.fillStyle = 'rgba(255, 255, 0, 0.6)';
            this.displayCtx.fillRect(currentTimeX - 1, 0, 2, destHeight);
        }

        // Draw time ruler (below the raster)
        this.drawTimeRuler(destHeight);
    }

    private drawTimeRuler(rasterHeight: number) {
        const { width, height } = this.display;
        const rulerHeight = 20;
        const rulerY = rasterHeight;
        
        // Background for ruler
        this.displayCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        this.displayCtx.fillRect(0, rulerY, width, rulerHeight);
        
        // Ticks and labels
        this.displayCtx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        this.displayCtx.fillStyle = 'white';
        this.displayCtx.font = '10px monospace';
        this.displayCtx.textAlign = 'center';
        
        // Calculate tick interval (aim for ~5-10 ticks)
        const visibleTime = this.windowSize;
        let tickInterval = 100; // ms
        if (visibleTime < 200) tickInterval = 20;
        else if (visibleTime < 500) tickInterval = 50;
        else if (visibleTime > 2000) tickInterval = 200;
        
        const startMs = Math.floor(this.windowStart);
        const endMs = Math.ceil(this.windowStart + this.windowSize);
        
        for (let t = Math.ceil(startMs / tickInterval) * tickInterval; t <= endMs; t += tickInterval) {
            const x = (t - this.windowStart) * (width / this.windowSize);
            if (x < 0 || x > width) continue;
            
            this.displayCtx.beginPath();
            this.displayCtx.moveTo(x, rulerY);
            this.displayCtx.lineTo(x, rulerY + rulerHeight);
            this.displayCtx.stroke();
            
            // Label in milliseconds
            this.displayCtx.fillText(`${t}`, x, rulerY + rulerHeight - 4);
        }
    }
}

const viz = new Visualizer();
viz.init().then(() => {
    console.log('Visualization ready');
}).catch(err => {
    console.error('Failed to initialize visualization:', err);
});
