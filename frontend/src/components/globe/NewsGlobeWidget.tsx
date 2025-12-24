/**
 * NEWS GLOBE WIDGET
 * 
 * A compact interactive 3D globe showing global news hotspots.
 * Designed to be embedded in the News page sidebar.
 * 
 * Features:
 * - Pulsing markers for news hotspots
 * - Auto-rotation with smooth easing
 * - Click to navigate to full Globe page
 * - Responsive sizing
 */

import { onMount, onCleanup, createSignal, Show } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ArrowRight, Globe } from 'lucide-solid';

const GLOBE_RADIUS = 50;

// Major financial centers with news activity
const NEWS_HOTSPOTS = [
  { name: 'New York', lat: 40.7128, lng: -74.0060, newsCount: 45, region: 'Americas' },
  { name: 'London', lat: 51.5074, lng: -0.1278, newsCount: 38, region: 'Europe' },
  { name: 'Tokyo', lat: 35.6762, lng: 139.6503, newsCount: 28, region: 'Asia' },
  { name: 'Hong Kong', lat: 22.3193, lng: 114.1694, newsCount: 24, region: 'Asia' },
  { name: 'Singapore', lat: 1.3521, lng: 103.8198, newsCount: 18, region: 'Asia' },
  { name: 'Frankfurt', lat: 50.1109, lng: 8.6821, newsCount: 15, region: 'Europe' },
  { name: 'Sydney', lat: -33.8688, lng: 151.2093, newsCount: 12, region: 'Pacific' },
  { name: 'Dubai', lat: 25.2048, lng: 55.2708, newsCount: 10, region: 'MENA' },
  { name: 'Shanghai', lat: 31.2304, lng: 121.4737, newsCount: 22, region: 'Asia' },
  { name: 'Mumbai', lat: 19.0760, lng: 72.8777, newsCount: 14, region: 'Asia' },
];

export default function NewsGlobeWidget() {
  const navigate = useNavigate();
  let containerRef: HTMLDivElement | undefined;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let renderer: THREE.WebGLRenderer;
  let controls: OrbitControls;
  let animationId: number;
  let globeGroup: THREE.Group;
  let cloudMesh: THREE.Mesh | null = null;
  const markerGroup = new THREE.Group();
  
  const [isHovered, setIsHovered] = createSignal(false);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_selectedHotspot, _setSelectedHotspot] = createSignal<typeof NEWS_HOTSPOTS[0] | null>(null);

  onMount(() => {
    if (!containerRef) return;
    init();
    animate();
    
    const handleResize = () => {
      if (!containerRef) return;
      camera.aspect = containerRef.clientWidth / containerRef.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.clientWidth, containerRef.clientHeight);
    };
    
    window.addEventListener('resize', handleResize);
    onCleanup(() => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationId);
      controls?.dispose();
      renderer?.dispose();
    });
  });

  function init() {
    if (!containerRef) return;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0f1a);

    // Camera
    camera = new THREE.PerspectiveCamera(
      45,
      containerRef.clientWidth / containerRef.clientHeight,
      1,
      500
    );
    camera.position.z = 140;

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(containerRef.clientWidth, containerRef.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerRef.appendChild(renderer.domElement);

    // Controls - simplified for widget
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enablePan = false;
    controls.enableZoom = false; // Disable zoom in widget
    controls.autoRotate = false; // We rotate the globe manually now
    
    // Globe Group (for tilt and rotation)
    globeGroup = new THREE.Group();
    scene.add(globeGroup);
    // Earth's axial tilt is approx 23.5 degrees
    globeGroup.rotation.z = 23.5 * Math.PI / 180;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xaaaaaa, 1);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 2.5);
    directionalLight.position.set(50, 50, 100);
    scene.add(directionalLight);

    // Globe
    createGlobe();
    
    // Markers
    globeGroup.add(markerGroup);
    createHotspotMarkers();

    // Stars
    createStars();
  }

  function createGlobe() {
    const globeGeometry = new THREE.SphereGeometry(GLOBE_RADIUS, 64, 64);
    const textureLoader = new THREE.TextureLoader();
    textureLoader.crossOrigin = 'anonymous';

    // High-fidelity earth textures (using reliable sources)
    const earthDayUrl = 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg';
    const earthBumpUrl = 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_normal_2048.jpg';
    const earthSpecUrl = 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_specular_2048.jpg';
    const earthNightUrl = 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_lights_2048.png';
    const earthCloudUrl = 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_clouds_1024.png';

    const earthMap = textureLoader.load(earthDayUrl, () => renderer.render(scene, camera));
    const earthBump = textureLoader.load(earthBumpUrl);
    const earthSpec = textureLoader.load(earthSpecUrl);
    const earthNight = textureLoader.load(earthNightUrl);
    const cloudMap = textureLoader.load(earthCloudUrl);

    const globeMaterial = new THREE.MeshPhongMaterial({
      map: earthMap,
      bumpMap: earthBump,
      bumpScale: 0.35,
      specularMap: earthSpec,
      specular: new THREE.Color(0x222222),
      shininess: 12,
      emissiveMap: earthNight,
      emissive: new THREE.Color(0x111111),
      emissiveIntensity: 0.35,
    });
    
    const globe = new THREE.Mesh(globeGeometry, globeMaterial);
    globeGroup.add(globe);

    // Soft cloud layer
    const cloudGeometry = new THREE.SphereGeometry(GLOBE_RADIUS * 1.01, 64, 64);
    const cloudMaterial = new THREE.MeshPhongMaterial({
      map: cloudMap,
      transparent: true,
      depthWrite: false,
      opacity: 0.35,
    });
    cloudMesh = new THREE.Mesh(cloudGeometry, cloudMaterial);
    globeGroup.add(cloudMesh);

    // Atmosphere
    createAtmosphere();
  }

  function createAtmosphere() {
    const atmosphereGeometry = new THREE.SphereGeometry(GLOBE_RADIUS * 1.15, 64, 64);
    const atmosphereMaterial = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        void main() {
          float intensity = pow(0.7 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
          vec3 innerGlow = vec3(0.298, 0.114, 0.584) * 0.4;
          vec3 outerGlow = vec3(0.114, 0.478, 0.871) * 0.3;
          vec3 atmosphere = mix(innerGlow, outerGlow, intensity);
          gl_FragColor = vec4(atmosphere, intensity * 0.5);
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
    });

    const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
    globeGroup.add(atmosphere);
  }

  function latLonToVector3(lat: number, lon: number, radius: number): THREE.Vector3 {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    return new THREE.Vector3(
      -(radius * Math.sin(phi) * Math.cos(theta)),
      radius * Math.cos(phi),
      radius * Math.sin(phi) * Math.sin(theta)
    );
  }

  function createHotspotMarkers() {
    NEWS_HOTSPOTS.forEach((hotspot, index) => {
      const position = latLonToVector3(hotspot.lat, hotspot.lng, GLOBE_RADIUS);
      
      // Size based on news count
      const size = 1.5 + (hotspot.newsCount / 45) * 2;
      
      // Marker core
      const coreGeometry = new THREE.SphereGeometry(size, 16, 16);
      const coreMaterial = new THREE.MeshBasicMaterial({
        color: 0x00d4ff,
        transparent: true,
        opacity: 0.9,
      });
      const core = new THREE.Mesh(coreGeometry, coreMaterial);
      core.position.copy(position);
      markerGroup.add(core);

      // Pulsing ring
      const ringGeometry = new THREE.RingGeometry(size * 1.2, size * 1.8, 32);
      const ringMaterial = new THREE.MeshBasicMaterial({
        color: 0x00d4ff,
        transparent: true,
        opacity: 0.4,
        side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(ringGeometry, ringMaterial);
      ring.position.copy(position);
      ring.lookAt(new THREE.Vector3(0, 0, 0));
      ring.userData = { 
        baseScale: 1, 
        phase: index * 0.5,
        type: 'ring',
        hotspot 
      };
      markerGroup.add(ring);
    });
  }

  function createStars() {
    const starGeometry = new THREE.BufferGeometry();
    const starCount = 500;
    const positions = new Float32Array(starCount * 3);
    
    for (let i = 0; i < starCount * 3; i += 3) {
      const radius = 200 + Math.random() * 100;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;
      
      positions[i] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i + 1] = radius * Math.cos(phi);
      positions[i + 2] = radius * Math.sin(phi) * Math.sin(theta);
    }
    
    starGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    const starMaterial = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.5,
      transparent: true,
      opacity: 0.6,
    });
    
    scene.add(new THREE.Points(starGeometry, starMaterial));
  }

  function animate() {
    animationId = requestAnimationFrame(animate);
    
    const time = Date.now() * 0.001;
    
    // Rotate the globe group (Earth rotation)
    if (globeGroup) {
      // Rotate West to East (counter-clockwise when viewed from North Pole)
      // Slow down on hover
      const rotationSpeed = isHovered() ? 0.0005 : 0.002;
      globeGroup.rotation.y += rotationSpeed;
    }

    // Rotate clouds slightly faster
    if (cloudMesh) {
      cloudMesh.rotation.y += 0.0005;
    }
    
    // Animate pulsing rings
    markerGroup.children.forEach((child) => {
      if (child.userData?.type === 'ring') {
        const phase = child.userData.phase || 0;
        const scale = 1 + 0.3 * Math.sin(time * 2 + phase);
        child.scale.set(scale, scale, scale);
        const mesh = child as THREE.Mesh;
        if (mesh.material && !Array.isArray(mesh.material)) {
          (mesh.material as THREE.MeshBasicMaterial).opacity = 0.4 * (1 - (scale - 1) / 0.3);
        }
      }
    });
    
    controls?.update();
    renderer?.render(scene, camera);
  }

  return (
    <div class="bg-terminal-900 border border-terminal-800 rounded-xl overflow-hidden">
      {/* Header */}
      <div class="p-3 border-b border-terminal-800 flex items-center justify-between">
        <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
          <Globe class="w-3.5 h-3.5 text-accent-400" />
          Global News Hotspots
        </h3>
        <button
          class="text-[10px] text-accent-400 hover:text-accent-300 transition-colors flex items-center gap-1"
          onClick={() => navigate('/globe')}
        >
          Expand
          <ArrowRight class="w-3 h-3" />
        </button>
      </div>
      
      {/* Globe Container */}
      <div 
        ref={containerRef}
        class="relative w-full h-48 cursor-grab active:cursor-grabbing"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        onClick={() => navigate('/globe')}
      >
        {/* Overlay hint */}
        <Show when={isHovered()}>
          <div class="absolute inset-0 flex items-center justify-center bg-black/30 pointer-events-none transition-opacity duration-300 z-10">
            <span class="text-xs text-accent-400 font-medium">Click to explore</span>
          </div>
        </Show>
      </div>
      
      {/* Hotspot Legend */}
      <div class="p-3 border-t border-terminal-800 bg-terminal-950/50">
        <div class="flex flex-wrap gap-1.5">
          <For each={NEWS_HOTSPOTS.slice(0, 5)}>
            {(hotspot) => (
              <div 
                class="flex items-center gap-1 px-2 py-1 bg-terminal-800/50 rounded text-[10px] hover:bg-accent-500/10 cursor-pointer transition-colors"
                onClick={() => navigate('/globe')}
              >
                <span class="w-1.5 h-1.5 rounded-full bg-accent-400 animate-pulse" />
                <span class="text-gray-400">{hotspot.name}</span>
                <span class="text-accent-400 font-bold">{hotspot.newsCount}</span>
              </div>
            )}
          </For>
        </div>
      </div>
    </div>
  );
}
