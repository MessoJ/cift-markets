/**
 * Interactive 3D Globe - Based on HTML Reference Design
 * 
 * Clean implementation with OrbitControls and TWEEN animations:
 * - Earth night texture
 * - Bright blue markers
 * - Smooth camera zoom on click
 * - OrbitControls for manual rotation
 * - Auto-rotation (toggleable via controls)
 * - Distance-based marker scaling
 * - Glassmorphism modal in center
 */

import { createSignal, onMount, onCleanup } from 'solid-js';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
// @ts-ignore - TWEEN doesn't have types
import TWEEN from 'https://cdn.jsdelivr.net/npm/@tweenjs/tween.js@23.1.1/dist/tween.esm.js';
import type { GlobeCountryData, GlobeNewsData } from '../../lib/api/client';

interface GlobalNewsGlobeProps {
  data: GlobeNewsData;
  autoRotate?: boolean;
  onCountryClick?: (country: GlobeCountryData) => void;
}

interface DataPoint {
  country: GlobeCountryData;
  mesh: THREE.Mesh;
  position: THREE.Vector3;
}

export function GlobalNewsGlobe(props: GlobalNewsGlobeProps) {
  let containerRef: HTMLDivElement | undefined;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let renderer: THREE.WebGLRenderer;
  let globe: THREE.Mesh;
  let controls: OrbitControls;
  let markerGroup: THREE.Group;
  let animationId: number;
  let raycaster: THREE.Raycaster;
  let mouse: THREE.Vector2;
  let dataPoints: DataPoint[] = [];
  let lastCameraPos = new THREE.Vector3(0, 0, 250);
  let intersectedMarker: THREE.Mesh | null = null;
  
  const GLOBE_RADIUS = 100;
  const MARKER_COLOR = 0x0088ff; // Bright blue
  const [selectedCountry, setSelectedCountry] = createSignal<GlobeCountryData | null>(null);

  onMount(() => {
    if (!containerRef) return;

    init();
    animate();

    // Event handlers
    const handleResize = () => {
      if (!containerRef) return;
      camera.aspect = containerRef.clientWidth / containerRef.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.clientWidth, containerRef.clientHeight);
    };

    const handleMouseMove = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(markerGroup.children);

      if (intersects.length > 0) {
        if (intersectedMarker !== intersects[0].object) {
          intersectedMarker = intersects[0].object as THREE.Mesh;
          document.body.style.cursor = 'pointer';
        }
      } else {
        if (intersectedMarker) {
          intersectedMarker = null;
          document.body.style.cursor = 'default';
        }
      }
    };

    const handleClick = () => {
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(markerGroup.children);

      if (intersects.length > 0) {
        const clickedObject = intersects[0].object;
        const clickedPoint = dataPoints.find(dp => dp.mesh === clickedObject);
        
        if (clickedPoint) {
          // Stop auto-rotation
          controls.autoRotate = false;
          
          // Store current camera position
          lastCameraPos.copy(camera.position);

          // Calculate target position
          const markerPos = clickedPoint.position;
          const cameraTargetPos = markerPos.clone().normalize().multiplyScalar(GLOBE_RADIUS + 30);
          
          // Clear any existing tweens
          TWEEN.removeAll();

          // Animate camera position
          new TWEEN.Tween(camera.position)
            .to(cameraTargetPos, 1000)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .start();

          // Animate camera target
          new TWEEN.Tween(controls.target)
            .to(markerPos, 1000)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onComplete(() => {
              setSelectedCountry(clickedPoint.country);
              props.onCountryClick?.(clickedPoint.country);
              controls.update();
            })
            .start();
        }
      }
    };

    window.addEventListener('resize', handleResize);
    renderer.domElement.addEventListener('mousemove', handleMouseMove, false);
    renderer.domElement.addEventListener('click', handleClick, false);

    onCleanup(() => {
      window.removeEventListener('resize', handleResize);
      renderer.domElement.removeEventListener('mousemove', handleMouseMove);
      renderer.domElement.removeEventListener('click', handleClick);
      
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
      
      TWEEN.removeAll();
      controls.dispose();
      renderer.dispose();
    });
  });

  function init() {
    if (!containerRef) return;

    // Scene setup - Dark space background
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x030014); // Deep space color

    // Camera setup
    camera = new THREE.PerspectiveCamera(
      45,
      containerRef.clientWidth / containerRef.clientHeight,
      1,
      1000
    );
    camera.position.z = 250;

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(containerRef.clientWidth, containerRef.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.appendChild(renderer.domElement);

    // OrbitControls (like reference)
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enablePan = false;
    controls.enableZoom = true;
    controls.minDistance = 105;
    controls.maxDistance = 400;
    controls.autoRotate = props.autoRotate ?? true;
    controls.autoRotateSpeed = 0.5;

    // Raycaster
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // Lights
    createLights();
    
    // Globe and markers
    createGlobe();
    createMarkers();
  }

  function createLights() {
    const ambientLight = new THREE.AmbientLight(0xaaaaaa, 1);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 2.5);
    directionalLight.position.set(100, 100, 200);
    scene.add(directionalLight);
  }

  function createGlobe() {
    const globeGeometry = new THREE.SphereGeometry(GLOBE_RADIUS, 64, 64);
    const textureLoader = new THREE.TextureLoader();
    
    const earthMap = textureLoader.load(
      'https://cdn.jsdelivr.net/npm/three-globe@2.31.0/example/img/earth-night.jpg',
      () => {
        console.log('Globe texture loaded successfully.');
        renderer.render(scene, camera);
      },
      undefined,
      () => {
        console.error('Error loading globe texture. Using fallback color.');
        if (globe) {
          globe.material = new THREE.MeshPhongMaterial({ color: 0x154289 });
        }
      }
    );

    const globeMaterial = new THREE.MeshPhongMaterial({
      map: earthMap,
      color: 0xffffff,
      shininess: 10,
    });

    globe = new THREE.Mesh(globeGeometry, globeMaterial);
    scene.add(globe);
  }

  function createMarkers() {
    markerGroup = new THREE.Group();
    
    const markerGeometry = new THREE.SphereGeometry(1.5, 16, 16);
    const markerMaterial = new THREE.MeshBasicMaterial({ color: MARKER_COLOR });

    props.data.countries.forEach(country => {
      const position = latLonToVector3(country.lat, country.lng, GLOBE_RADIUS + 0.5);
      const marker = new THREE.Mesh(markerGeometry, markerMaterial);
      marker.position.copy(position);
      
      // Store data in marker
      marker.userData = country;
      
      markerGroup.add(marker);
      
      dataPoints.push({
        country,
        mesh: marker,
        position,
      });
    });

    scene.add(markerGroup);
  }

  function latLonToVector3(lat: number, lng: number, radius: number): THREE.Vector3 {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lng + 180) * (Math.PI / 180);

    const x = -(radius * Math.sin(phi) * Math.cos(theta));
    const z = radius * Math.sin(phi) * Math.sin(theta);
    const y = radius * Math.cos(phi);

    return new THREE.Vector3(x, y, z);
  }

  function animate() {
    animationId = requestAnimationFrame(animate);
    
    TWEEN.update(); // Update camera animations
    controls.update(); // Update OrbitControls

    // Distance-based marker scaling (like reference)
    if (markerGroup) {
      markerGroup.children.forEach(marker => {
        const distance = camera.position.distanceTo(marker.position);
        const scale = distance / 500;
        marker.scale.set(scale, scale, scale);
      });
    }

    renderer.render(scene, camera);
  }

  function hideModal() {
    setSelectedCountry(null);

    // Animate camera back
    TWEEN.removeAll();

    new TWEEN.Tween(camera.position)
      .to(lastCameraPos, 1000)
      .easing(TWEEN.Easing.Quadratic.InOut)
      .start();

    new TWEEN.Tween(controls.target)
      .to(new THREE.Vector3(0, 0, 0), 1000)
      .easing(TWEEN.Easing.Quadratic.InOut)
      .onComplete(() => {
        controls.autoRotate = true;
        controls.update();
      })
      .start();
  }

  return (
    <div class="relative w-full h-full">
      {/* Canvas Container */}
      <div ref={containerRef} class="w-full h-full" />

      {/* Modal (like reference) */}
      {selectedCountry() && (
        <div 
          class="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[300px] bg-[rgba(10,10,20,0.8)] backdrop-blur-[10px] border border-white/10 rounded-[20px] p-6 shadow-[0_10px_30px_rgba(0,0,0,0.3)] z-[100] animate-fade-in"
          role="dialog"
          aria-labelledby="modal-title"
        >
          {/* Close button */}
          <button
            onClick={hideModal}
            class="absolute top-4 right-4 text-2xl text-[#aaa] hover:text-white bg-none border-none p-0 cursor-pointer leading-none transition-colors"
            aria-label="Close dialog"
          >
            ×
          </button>

          {/* Header */}
          <div class="flex items-center gap-4 mb-4">
            <div class="w-[60px] h-[60px] rounded-xl bg-white overflow-hidden flex items-center justify-center">
              <div class="text-2xl font-bold text-gray-800">
                {selectedCountry()!.code}
              </div>
            </div>
            <div>
              <h2 id="modal-title" class="text-xl font-semibold leading-tight text-white">
                {selectedCountry()!.name}
              </h2>
            </div>
          </div>

          {/* Subtitle */}
          <p class="text-base text-[#aaa] mb-5">
            {selectedCountry()!.region} • {selectedCountry()!.article_count} Articles
          </p>

          {/* Location */}
          <div class="flex items-center gap-3 text-base font-medium text-white">
            <span class="text-2xl">{selectedCountry()!.code === 'US' ? '🇺🇸' : '🌍'}</span>
            <span>{selectedCountry()!.name}</span>
          </div>
        </div>
      )}
    </div>
  );
}
