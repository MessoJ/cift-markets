/**
 * Enhanced Financial Globe with Stock Exchanges, Arcs, and Boundaries
 * 
 * Features:
 * - Stock exchange markers (from API)
 * - Animated news arcs between markets
 * - Political boundaries with sentiment coloring
 * - Search and filtering
 * - Click-to-zoom interactions
 * - Distance-based marker scaling
 */
import { createSignal, createEffect, onMount, onCleanup, For, Show } from 'solid-js';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
// @ts-ignore - TWEEN doesn't have types
import TWEEN from 'https://cdn.jsdelivr.net/npm/@tweenjs/tween.js@23.1.1/dist/tween.esm.js';
import { useGlobeData, type GlobeExchange, type GlobeArc } from '../../hooks/useGlobeData';
import { useAssetData, type AssetLocation } from '../../hooks/useAssetData';
import { useShipData, type TrackedShip } from '../../hooks/useShipData';
import { GlobeFilterPanel, type GlobeFilters } from './GlobeFilterPanel';
import { type CountryData } from './CountryModal';
import { GlobeSearch, type SearchResult } from './GlobeSearch';
import { COUNTRY_CAPITALS, type CapitalData } from '../../data/countryCapitals';
import { 
  AssetMarkerData, 
  AssetCategory, 
  EventStatus,
  ASSET_COLORS, 
  EVENT_STATUS_COLORS,
  calculateEventStatus,
  getMarkerSizeMultiplier,
  getPulseSpeed
} from '../../config/assetColors';
import { assetWebSocket, useAssetStream } from '../../services/assetWebSocket';
import { AssetDetailModal } from './AssetDetailModal';

interface EnhancedFinancialGlobeProps {
  autoRotate?: boolean;
  showArcs?: boolean;
  showBoundaries?: boolean;
  showAssets?: boolean;
  onExchangeClick?: (exchange: GlobeExchange) => void;
}

interface MarkerData {
  exchange: GlobeExchange;
  mesh: THREE.Mesh;
  position: THREE.Vector3;
}

interface AssetMarkerData {
  asset: AssetLocation;
  mesh: THREE.Mesh;
  position: THREE.Vector3;
}

interface ShipMarkerData {
  ship: TrackedShip;
  mesh: THREE.Mesh;
  position: THREE.Vector3;
  trail: THREE.Line | null;
}

interface CapitalMarkerData {
  capital: CapitalData;
  mesh: THREE.Mesh;
  position: THREE.Vector3;
}

export function EnhancedFinancialGlobe(props: EnhancedFinancialGlobeProps) {
  console.log('🌟 EnhancedFinancialGlobe component rendered');
  
  let containerRef: HTMLDivElement | undefined;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let renderer: THREE.WebGLRenderer;
  let globe: THREE.Mesh;
  let globeGroup: THREE.Group;
  let controls: OrbitControls;
  let markerGroup: THREE.Group;
  let markers: MarkerData[] = [];
  let assetMarkerGroup: THREE.Group;
  let assetMarkers: AssetMarkerData[] = [];
  let shipMarkerGroup: THREE.Group;
  let shipMarkers: ShipMarkerData[] = [];
  let capitalMarkerGroup: THREE.Group;
  let capitalMarkers: CapitalMarkerData[] = [];
  let arcGroup: THREE.Group;
  let arcLines: THREE.Line[] = [];
  let boundaryGroup: THREE.Group;
  let boundaryLines: THREE.Line[] = [];
  let animationId: number;
  let cloudMesh: THREE.Mesh | null = null;
  let raycaster: THREE.Raycaster;
  let mouse: THREE.Vector2;
  let lastCameraPos = new THREE.Vector3(0, 0, 250);
  let intersectedMarker: THREE.Mesh | null = null;
  let intersectedAssetMarker: THREE.Mesh | null = null;
  let intersectedCapitalMarker: THREE.Mesh | null = null;
  
  const GLOBE_RADIUS = 100;
  const MARKER_BASE_SIZE = 0.8;
  const ASSET_BASE_SIZE = 1.0;
  const [selectedAsset, setSelectedAsset] = createSignal<AssetLocation | null>(null);
  const [selectedAssetMarker, setSelectedAssetMarker] = createSignal<AssetMarkerData | null>(null);
  const [selectedCountry, setSelectedCountry] = createSignal<CountryData | null>(null);
  const [selectedCapital, setSelectedCapital] = createSignal<CapitalData | null>(null);
  const [hoveredExchange, setHoveredExchange] = createSignal<GlobeExchange | null>(null);
  const [hoveredAsset, setHoveredAsset] = createSignal<AssetLocation | null>(null);
  const [hoveredCapital, setHoveredCapital] = createSignal<CapitalData | null>(null);

  // Real-time asset stream
  const { assets: liveAssets, latestNews, events } = useAssetStream();

  // Connect WebSocket on mount
  onMount(() => {
    assetWebSocket.connect();
  });

  // Prepare search data
  const searchData = (): SearchResult[] => {
    const results: SearchResult[] = [];

    // Add exchanges
    const exchangeData = exchanges();
    if (exchangeData) {
      exchangeData.forEach(ex => {
        results.push({
          type: 'exchange',
          name: ex.name,
          code: ex.code,
          lat: ex.lat,
          lng: ex.lng,
          subtitle: ex.country,
          flag: ex.flag,
        });
      });
    }

    // Add assets
    const assetData = assets();
    if (assetData) {
      assetData.forEach(asset => {
        results.push({
          type: 'asset',
          name: asset.name,
          code: asset.code,
          lat: asset.lat,
          lng: asset.lng,
          subtitle: `${asset.asset_type.replace('_', ' ')} • ${asset.city}, ${asset.country}`,
          flag: asset.flag,
        });
      });
    }

    // Add ships
    const shipData = ships();
    if (shipData) {
      shipData.forEach(ship => {
        results.push({
          type: 'ship',
          name: ship.ship_name,
          code: ship.mmsi,
          lat: ship.current_lat,
          lng: ship.current_lng,
          subtitle: `${ship.ship_type.replace('_', ' ')} • ${ship.flag_country}`,
          flag: ship.flag_country_code ? getCountryFlag(ship.flag_country_code) : '🚢',
        });
      });
    }

    return results;
  };
  
  // Filter state
  const [filters, setFilters] = createSignal<GlobeFilters>({
    showExchanges: true,
    showAssets: props.showAssets ?? true,
    showArcs: props.showArcs ?? true,
    showBoundaries: props.showBoundaries ?? true,
    assetTypes: {
      central_bank: true,
      commodity_market: true,
      government: true,
      tech_hq: true,
      energy: true,
    },
    assetStatus: {
      operational: true,
      unknown: true,
      issue: true,
    },
  });

  console.log('📞 Calling useGlobeData hook...');
  // Use globe data hook - Show all exchanges regardless of news count
  const { exchanges, arcs, boundaries, loading, error } = useGlobeData({
    timeframe: '24h',
    min_articles: 0, // Changed from 1 to show all 40 exchanges
    min_strength: 0.3,
  });
  console.log('✅ useGlobeData returned:', { 
    exchangesFunction: typeof exchanges,
    hasExchanges: exchanges()?.length || 0
  });

  console.log('📞 Calling useAssetData hook...');
  // Use asset data hook - Show assets if enabled
  const { 
    assets, 
    statusSummary, 
    loading: assetsLoading, 
    error: assetsError 
  } = useAssetData({
    timeframe: '24h',
    asset_type: 'all',
    status: 'all',
    min_importance: 0,
  });
  console.log('✅ useAssetData returned:', {
    assetsFunction: typeof assets,
    hasAssets: assets()?.length || 0
  });

  console.log('📞 Calling useShipData hook...');
  // Use ship data hook
  const {
    ships,
    loading: shipsLoading,
    error: shipsError
  } = useShipData({
    ship_type: 'all',
    min_importance: 0,
  });
  console.log('✅ useShipData returned:', {
    shipsFunction: typeof ships,
    hasShips: ships()?.length || 0
  });

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
      
      // Check exchanges first
      const intersects = raycaster.intersectObjects(markerGroup.children);
      // Then check assets
      const intersectsAssets = raycaster.intersectObjects(assetMarkerGroup.children);
      // Then check capitals
      const intersectsCapitals = raycaster.intersectObjects(capitalMarkerGroup.children);

      if (intersects.length > 0) {
        if (intersectedMarker !== intersects[0].object) {
          intersectedMarker = intersects[0].object as THREE.Mesh;
          intersectedAssetMarker = null;
          intersectedCapitalMarker = null;
          document.body.style.cursor = 'pointer';
          
          // Find and set hovered exchange
          const hoveredMarker = markers.find(m => m.mesh === intersectedMarker);
          if (hoveredMarker) {
            setHoveredExchange(hoveredMarker.exchange);
            setHoveredAsset(null);
            setHoveredCapital(null);
          }
        }
      } else if (intersectsAssets.length > 0) {
        if (intersectedAssetMarker !== intersectsAssets[0].object) {
          intersectedAssetMarker = intersectsAssets[0].object as THREE.Mesh;
          intersectedMarker = null;
          intersectedCapitalMarker = null;
          document.body.style.cursor = 'pointer';
          
          // Find and set hovered asset
          const hoveredAssetMarker = assetMarkers.find(m => m.mesh === intersectedAssetMarker);
          if (hoveredAssetMarker) {
            setHoveredAsset(hoveredAssetMarker.asset);
            setHoveredExchange(null);
            setHoveredCapital(null);
          }
        }
      } else if (intersectsCapitals.length > 0) {
        if (intersectedCapitalMarker !== intersectsCapitals[0].object) {
          intersectedCapitalMarker = intersectsCapitals[0].object as THREE.Mesh;
          intersectedMarker = null;
          intersectedAssetMarker = null;
          document.body.style.cursor = 'pointer';
          
          // Find and set hovered capital
          const hoveredCapitalMarker = capitalMarkers.find(m => m.mesh === intersectedCapitalMarker);
          if (hoveredCapitalMarker) {
            setHoveredCapital(hoveredCapitalMarker.capital);
            setHoveredExchange(null);
            setHoveredAsset(null);
          }
        }
      } else {
        if (intersectedMarker || intersectedAssetMarker || intersectedCapitalMarker) {
          intersectedMarker = null;
          intersectedAssetMarker = null;
          intersectedCapitalMarker = null;
          document.body.style.cursor = 'default';
          setHoveredExchange(null);
          setHoveredAsset(null);
          setHoveredCapital(null);
        }
      }
    };

    const handleClick = () => {
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(markerGroup.children);
      const intersectsAssets = raycaster.intersectObjects(assetMarkerGroup.children);
      const intersectsCapitals = raycaster.intersectObjects(capitalMarkerGroup.children);

      // Priority: Exchanges > Assets > Capitals
      if (intersects.length > 0) {
        const clickedObject = intersects[0].object;
        const clickedMarker = markers.find(m => m.mesh === clickedObject);
        
        if (clickedMarker) {
          // Stop auto-rotation
          controls.autoRotate = false;
          
          // Store current camera position
          lastCameraPos.copy(camera.position);

          // Calculate target position
          const markerPos = clickedMarker.position;
          const cameraTargetPos = markerPos.clone().normalize().multiplyScalar(GLOBE_RADIUS + 50);
          
          // Clear any existing tweens
          TWEEN.removeAll();

          // Animate camera position with smooth easing
          new TWEEN.Tween(camera.position)
            .to(cameraTargetPos, 1500)
            .easing(TWEEN.Easing.Cubic.InOut)
            .start();

          // Animate camera target
          new TWEEN.Tween(controls.target)
            .to(markerPos, 1500)
            .easing(TWEEN.Easing.Cubic.InOut)
            .onUpdate(() => {
              controls.update();
            })
            .onComplete(() => {
              // Delay modal slightly for smoother feel
              setTimeout(() => {
                props.onExchangeClick?.(clickedMarker.exchange);
              }, 100);
              controls.update();
            })
            .start();
        }
      } else if (intersectsAssets.length > 0) {
        const clickedObject = intersectsAssets[0].object;
        const clickedAsset = assetMarkers.find(m => m.mesh === clickedObject);
        
        if (clickedAsset) {
          // Stop auto-rotation
          controls.autoRotate = false;
          
          // Store current camera position
          lastCameraPos.copy(camera.position);

          // Calculate target position
          const assetPos = clickedAsset.position;
          const cameraTargetPos = assetPos.clone().normalize().multiplyScalar(GLOBE_RADIUS + 50);
          
          // Clear any existing tweens
          TWEEN.removeAll();

          // Animate camera position with smooth easing
          new TWEEN.Tween(camera.position)
            .to(cameraTargetPos, 1500)
            .easing(TWEEN.Easing.Cubic.InOut)
            .start();

          // Animate camera target
          new TWEEN.Tween(controls.target)
            .to(assetPos, 1500)
            .easing(TWEEN.Easing.Cubic.InOut)
            .onUpdate(() => {
              controls.update();
            })
            .onComplete(() => {
              // Delay modal slightly for smoother feel
              setTimeout(() => {
                // Use the new AssetMarkerData structure stored in userData
                const assetData = clickedObject.userData.asset as AssetMarkerData;
                setSelectedAssetMarker(assetData);
              }, 100);
              controls.update();
            })
            .start();
        }
      } else if (intersectsCapitals.length > 0) {
        // Capital marker clicked
        const clickedObject = intersectsCapitals[0].object;
        const clickedCapital = capitalMarkers.find(m => m.mesh === clickedObject);
        
        if (clickedCapital) {
          // Stop auto-rotation
          controls.autoRotate = false;
          
          // Store current camera position
          lastCameraPos.copy(camera.position);

          // Calculate target position
          const capitalPos = clickedCapital.position;
          const cameraTargetPos = capitalPos.clone().normalize().multiplyScalar(GLOBE_RADIUS + 50);
          
          // Clear any existing tweens
          TWEEN.removeAll();

          // Animate camera position with smooth easing
          new TWEEN.Tween(camera.position)
            .to(cameraTargetPos, 1500)
            .easing(TWEEN.Easing.Cubic.InOut)
            .start();

          // Animate camera target
          new TWEEN.Tween(controls.target)
            .to(capitalPos, 1500)
            .easing(TWEEN.Easing.Cubic.InOut)
            .onUpdate(() => {
              controls.update();
            })
            .onComplete(() => {
              // Delay modal slightly for smoother feel
              setTimeout(() => {
                fetchCountryDetails(clickedCapital.capital.countryCode);
              }, 100);
              controls.update();
            })
            .start();
        }
      }
    };

    window.addEventListener('resize', handleResize);
    renderer.domElement.addEventListener('mousemove', handleMouseMove, false);
    renderer.domElement.addEventListener('click', handleClick, false);
  });

  // React to data changes
  createEffect(() => {
    const exchangeData = exchanges();
    const currentFilters = filters();
    console.log('🔥 createEffect triggered - exchanges:', exchangeData?.length || 0, 'showExchanges:', currentFilters.showExchanges);
    if (exchangeData && exchangeData.length > 0 && markerGroup && currentFilters.showExchanges) {
      console.log('✨ Creating markers for', exchangeData.length, 'exchanges');
      updateMarkers();
    } else if (markerGroup && !currentFilters.showExchanges) {
      markerGroup.clear();
      markers = [];
      console.log('🧹 Exchange markers cleared (showExchanges=false)');
    }
  });

  createEffect(() => {
    const arcData = arcs();
    const currentFilters = filters();
    if (arcData && arcData.length > 0 && arcGroup && currentFilters.showArcs) {
      console.log('Creating arcs for', arcData.length, 'connections');
      updateArcs();
    } else if (arcGroup && !currentFilters.showArcs) {
      arcGroup.clear();
      arcLines = [];
    }
  });

  createEffect(() => {
    const assetData = assets();
    const currentFilters = filters();
    console.log('🔥 createEffect triggered - assets:', assetData?.length || 0, 'showAssets:', currentFilters.showAssets);
    if (assetData && assetData.length > 0 && assetMarkerGroup && currentFilters.showAssets) {
      console.log('✨ Creating asset markers for', assetData.length, 'assets');
      updateAssetMarkers();
    } else if (assetMarkerGroup && !currentFilters.showAssets) {
      // Clear asset markers if showAssets is false
      assetMarkerGroup.clear();
      assetMarkers = [];
      console.log('🧹 Asset markers cleared (showAssets=false)');
    }
  });

  createEffect(() => {
    const boundaryData = boundaries();
    const currentFilters = filters();
    if (boundaryData && boundaryData.length > 0 && boundaryGroup && currentFilters.showBoundaries) {
      console.log('✨ Creating boundaries for', boundaryData.length, 'countries');
      updateBoundaries();
    } else if (boundaryGroup && !currentFilters.showBoundaries) {
      boundaryGroup.clear();
      boundaryLines = [];
      console.log('🧹 Boundaries cleared (showBoundaries=false)');
    }
  });

  // React to ship data changes
  createEffect(() => {
    const shipData = ships();
    if (shipData && shipData.length > 0 && shipMarkerGroup) {
      console.log('🚢 Ship data received, creating markers for', shipData.length, 'ships');
      updateShipMarkers();
    } else if (shipMarkerGroup) {
      shipMarkerGroup.clear();
      shipMarkers = [];
      console.log('🧹 Ship markers cleared');
    }
  });

  // Create capital markers on mount
  createEffect(() => {
    if (capitalMarkerGroup) {
      console.log('🏛️ Creating capital city markers...');
      updateCapitalMarkers();
    }
  });

  onCleanup(() => {
    if (containerRef) {
      window.removeEventListener('resize', () => {});
      renderer.domElement.removeEventListener('mousemove', () => {});
      renderer.domElement.removeEventListener('click', () => {});
    }
    
    if (animationId) {
      cancelAnimationFrame(animationId);
    }
    
    TWEEN.removeAll();
    if (controls) controls.dispose();
    if (renderer) renderer.dispose();
    
    // Disconnect WebSocket
    assetWebSocket.disconnect();
  });

  function init() {
    if (!containerRef) return;

    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x030014);

    // Camera setup
    camera = new THREE.PerspectiveCamera(
      45,
      containerRef.clientWidth / containerRef.clientHeight,
      1,
      1000
    );
    camera.position.z = 280; // Increased from 250 to fix globe cutoff

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(containerRef.clientWidth, containerRef.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.appendChild(renderer.domElement);

    // OrbitControls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enablePan = false;
    controls.enableZoom = true;
    controls.minDistance = 40; // Allow deeper zoom for earth detail
    controls.maxDistance = 600; // Allow far zoom for full view
    // Disable OrbitControls autoRotate since we are rotating the globe mesh manually
    controls.autoRotate = false; 
    controls.autoRotateSpeed = 0.25; // Slower rotation for better viewing

    // Raycaster
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // Globe Group (for tilt and rotation)
    globeGroup = new THREE.Group();
    scene.add(globeGroup);
    // Earth's axial tilt is approx 23.5 degrees
    globeGroup.rotation.z = 23.5 * Math.PI / 180;

    // Groups
    markerGroup = new THREE.Group();
    assetMarkerGroup = new THREE.Group();
    shipMarkerGroup = new THREE.Group();
    capitalMarkerGroup = new THREE.Group();
    arcGroup = new THREE.Group();
    boundaryGroup = new THREE.Group();
    
    // Add all data layers to the globe group so they rotate/tilt with it
    globeGroup.add(markerGroup);
    globeGroup.add(assetMarkerGroup);
    globeGroup.add(shipMarkerGroup);
    globeGroup.add(capitalMarkerGroup);
    globeGroup.add(arcGroup);
    globeGroup.add(boundaryGroup);

    // Create scene elements
    createLights();
    createGlobe();
    createStars();
  }

  function createLights() {
    const ambientLight = new THREE.AmbientLight(0xaaaaaa, 1);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 2.5);
    directionalLight.position.set(100, 100, 200);
    scene.add(directionalLight);
  }

  function createGlobe() {
    const globeGeometry = new THREE.SphereGeometry(GLOBE_RADIUS, 128, 128);
    const textureLoader = new THREE.TextureLoader();
    textureLoader.crossOrigin = 'anonymous';

    // Higher-fidelity earth textures for a more realistic globe (using reliable sources)
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

    globe = new THREE.Mesh(globeGeometry, globeMaterial);
    globeGroup.add(globe);

    // Soft cloud layer adds depth without heavy geometry cost
    const cloudGeometry = new THREE.SphereGeometry(GLOBE_RADIUS * 1.01, 96, 96);
    const cloudMaterial = new THREE.MeshPhongMaterial({
      map: cloudMap,
      transparent: true,
      depthWrite: false,
      opacity: 0.35,
    });
    cloudMesh = new THREE.Mesh(cloudGeometry, cloudMaterial);
    globeGroup.add(cloudMesh);

    // Add atmospheric glow
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

  function createStars() {
    const starGeometry = new THREE.BufferGeometry();
    const starCount = 1000;
    const positions = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 2000;
      positions[i + 1] = (Math.random() - 0.5) * 2000;
      positions[i + 2] = (Math.random() - 0.5) * 2000;
    }

    starGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const starMaterial = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 1,
      transparent: true,
      opacity: 0.8,
    });

    const stars = new THREE.Points(starGeometry, starMaterial);
    scene.add(stars);
  }

  function updateMarkers() {
    // Clear existing markers
    markerGroup.clear();
    markers = [];

    const exchangeData = exchanges();
    if (!exchangeData || exchangeData.length === 0) {
      console.warn('No exchange data to display');
      return;
    }

    console.log(`📍 Creating ${exchangeData.length} exchange markers...`);

    exchangeData.forEach((exchange: GlobeExchange) => {
      const position = latLonToVector3(exchange.lat, exchange.lng, GLOBE_RADIUS + 0.5);
      
      // Marker size based on news count (logarithmic scale)
      const sizeMultiplier = Math.log10(Math.max(exchange.news_count, 1) + 1) + 1;
      const markerSize = MARKER_BASE_SIZE * sizeMultiplier;
      
      const markerGeometry = new THREE.SphereGeometry(markerSize, 16, 16);
      
      // Use category color for exchanges
      const categoryColor = ASSET_COLORS[AssetCategory.EXCHANGE];
      let color = new THREE.Color(categoryColor);
      
      // Override with event status color if active event
      const eventStatus = calculateEventStatus(exchange.sentiment_score * 10, undefined, false);
      if (eventStatus !== EventStatus.NORMAL) {
        color = new THREE.Color(EVENT_STATUS_COLORS[eventStatus]);
      }
      
      const markerMaterial = new THREE.MeshBasicMaterial({ 
        color,
        transparent: true,
        opacity: 0.9,
      });

      const marker = new THREE.Mesh(markerGeometry, markerMaterial);
      marker.position.copy(position);
      marker.userData = { 
        exchange, 
        originalSize: markerSize,
        eventStatus,
        pulseSpeed: getPulseSpeed(eventStatus),
        pulsePhase: Math.random() * Math.PI * 2
      };
      
      markerGroup.add(marker);
      
      markers.push({
        exchange,
        mesh: marker,
        position,
      });
    });

    console.log(`✅ Added ${markers.length} markers to scene`);
    console.log('Sample marker:', markers[0]?.exchange?.name, markers[0]?.exchange?.code);
  }

  function updateCapitalMarkers() {
    // Clear existing capital markers
    capitalMarkerGroup.clear();
    capitalMarkers = [];

    console.log(`🏛️ Creating ${COUNTRY_CAPITALS.length} capital city markers...`);

    COUNTRY_CAPITALS.forEach((capital: CapitalData) => {
      const position = latLonToVector3(capital.lat, capital.lng, GLOBE_RADIUS + 0.3);
      
      // Smaller, duller markers compared to exchanges
      const markerSize = MARKER_BASE_SIZE * 0.6;
      
      const markerGeometry = new THREE.SphereGeometry(markerSize, 12, 12);
      
      // Dull grey-blue color for capitals
      const color = 0x6688aa;
      
      const markerMaterial = new THREE.MeshBasicMaterial({ 
        color,
        transparent: true,
        opacity: 0.7, // More transparent than exchanges
      });

      const marker = new THREE.Mesh(markerGeometry, markerMaterial);
      marker.position.copy(position);
      marker.userData = { capital, originalSize: markerSize, type: 'capital' };
      
      capitalMarkerGroup.add(marker);
      
      capitalMarkers.push({
        capital,
        mesh: marker,
        position,
      });
    });

    console.log(`✅ Added ${capitalMarkers.length} capital markers to scene`);
  }

  function updateAssetMarkers() {
    if (!props.showAssets) return;

    // Clear existing asset markers
    assetMarkerGroup.clear();
    assetMarkers = [];

    const assetData = assets();
    if (!assetData || assetData.length === 0) {
      console.log('ℹ️ No asset data to display');
      return;
    }

    // Filter assets by type and status
    const currentFilters = filters();
    const filteredAssets = assetData.filter((asset: AssetLocation) => {
      const typeEnabled = currentFilters.assetTypes[asset.asset_type];
      const statusEnabled = currentFilters.assetStatus[asset.current_status];
      return typeEnabled && statusEnabled;
    });

    console.log(`🏛️ Creating ${filteredAssets.length} asset markers (filtered from ${assetData.length})...`);

    filteredAssets.forEach((asset: AssetLocation) => {
      const position = latLonToVector3(asset.lat, asset.lng, GLOBE_RADIUS + 3.5);
      
      // Size based on importance score (0-100)
      const sizeMultiplier = (asset.importance_score / 50) * 0.8 + 0.5;
      const markerSize = ASSET_BASE_SIZE * sizeMultiplier;
      
      // Create geometry based on asset type
      let geometry: THREE.BufferGeometry;
      switch (asset.asset_type) {
        case 'central_bank':
          geometry = new THREE.BoxGeometry(markerSize, markerSize, markerSize);
          break;
        case 'commodity_market':
          geometry = new THREE.CylinderGeometry(markerSize * 0.6, markerSize * 0.6, markerSize * 1.2, 8);
          break;
        case 'government':
          geometry = new THREE.TetrahedronGeometry(markerSize * 0.8);
          break;
        case 'tech_hq':
          geometry = new THREE.OctahedronGeometry(markerSize * 0.8);
          break;
        case 'energy':
          geometry = new THREE.ConeGeometry(markerSize * 0.7, markerSize * 1.4, 8);
          break;
        default:
          geometry = new THREE.SphereGeometry(markerSize * 0.8, 8, 8);
      }
      
      // Map legacy types to new categories for color
      let category = AssetCategory.UNKNOWN;
      switch (asset.asset_type) {
        case 'stock_exchange': category = AssetCategory.EXCHANGE; break;
        case 'commodity_market': category = AssetCategory.COMMODITY; break;
        case 'forex_market': category = AssetCategory.FOREX; break;
        case 'crypto_exchange': category = AssetCategory.CRYPTO; break;
        case 'central_bank': category = AssetCategory.CAPITAL; break;
        case 'tech_hq': category = AssetCategory.EQUITY; break;
        case 'energy': category = AssetCategory.COMMODITY; break;
        case 'manufacturing': category = AssetCategory.EQUITY; break;
        case 'government': category = AssetCategory.BOND; break;
      }

      // Map status to event status
      let eventStatus = EventStatus.NORMAL;
      if (asset.current_status === 'issue') eventStatus = EventStatus.WARNING;
      if (asset.importance_score > 90) eventStatus = EventStatus.CRITICAL;

      // Get colors
      const baseColor = ASSET_COLORS[category] || '#888888';
      let color = new THREE.Color(baseColor);
      
      // Override with event status if critical/warning
      if (eventStatus !== EventStatus.NORMAL) {
        color = new THREE.Color(EVENT_STATUS_COLORS[eventStatus]);
      }
      
      const material = new THREE.MeshPhongMaterial({ 
        color,
        transparent: true,
        opacity: 0.9,
        emissive: color,
        emissiveIntensity: 0.3,
        shininess: 30,
      });

      const marker = new THREE.Mesh(geometry, material);
      marker.position.copy(position);
      
      // Store data for interaction and animation
      const assetMarkerData: AssetMarkerData = {
        id: asset.id,
        name: asset.name,
        category,
        lat: asset.lat,
        lng: asset.lng,
        eventStatus,
        lastUpdate: new Date(),
        value: asset.importance_score * 1000000000, // Mock value based on importance
      };

      marker.userData = { 
        asset: assetMarkerData, // Store the new format
        originalAsset: asset,   // Store original for reference
        originalSize: markerSize,
        pulseSpeed: getPulseSpeed(eventStatus),
        pulsePhase: Math.random() * Math.PI * 2
      };
      
      assetMarkerGroup.add(marker);
      
      assetMarkers.push({
        asset,
        mesh: marker,
        position,
      });
    });

    console.log(`✅ Added ${assetMarkers.length} asset markers to scene`);
  }

  function updateShipMarkers() {
    // Clear existing ship markers
    shipMarkerGroup.clear();
    shipMarkers = [];

    const shipData = ships();
    if (!shipData || shipData.length === 0) {
      console.log('ℹ️ No ship data to display');
      return;
    }

    console.log(`🚢 Creating ${shipData.length} ship markers with directional arrows...`);

    shipData.forEach((ship: TrackedShip) => {
      const position = latLonToVector3(ship.current_lat, ship.current_lng, GLOBE_RADIUS + 2);
      
      // Size based on importance
      const baseSize = 1.5;
      const sizeMultiplier = (ship.importance_score / 50) * 0.6 + 0.6;
      const markerSize = baseSize * sizeMultiplier;
      
      // Create directional arrow geometry (cone pointing forward)
      const geometry = new THREE.ConeGeometry(
        markerSize * 0.4,    // radius
        markerSize * 1.2,    // height (longer for arrow look)
        8                     // segments
      );
      
      // Color coding based on status
      let color: number;
      let statusText: string;
      
      // Check for issues in status
      const status = ship.current_status?.toLowerCase() || 'unknown';
      const hasIssue = status.includes('issue') || 
                       status.includes('problem') || 
                       status.includes('alert') ||
                       status.includes('danger') ||
                       status.includes('emergency');
      const isOperational = status.includes('operational') || 
                           status.includes('normal') ||
                           status.includes('active');
      
      if (hasIssue) {
        color = 0xff0000;    // Red for major issues
        statusText = 'Issues';
      } else if (isOperational) {
        color = 0x00ff00;    // Green for no issues
        statusText = 'Operational';
      } else {
        color = 0x808080;    // Grey for unknown status
        statusText = 'Unknown';
      }
      
      const material = new THREE.MeshPhongMaterial({ 
        color,
        transparent: true,
        opacity: 0.9,
        emissive: color,
        emissiveIntensity: 0.4,
        shininess: 50,
      });

      const marker = new THREE.Mesh(geometry, material);
      marker.position.copy(position);
      marker.userData = { ship, originalSize: markerSize, type: 'ship', statusText };
      
      // Orient marker to point outward from globe surface first
      marker.lookAt(new THREE.Vector3(0, 0, 0));
      marker.rotateX(Math.PI / 2); // Point tip outward
      
      // Rotate around the local Z axis based on ship's course
      // current_course is in degrees (0-360), where 0 is North
      if (ship.current_course !== null && ship.current_course !== undefined) {
        const courseRadians = THREE.MathUtils.degToRad(ship.current_course);
        marker.rotateZ(-courseRadians); // Negative because of coordinate system
      }
      
      shipMarkerGroup.add(marker);
      
      shipMarkers.push({
        ship,
        mesh: marker,
        position,
        trail: null,
      });
    });

    console.log(`✅ Added ${shipMarkers.length} ship arrow markers to scene`);
  }

  function updateArcs() {
    if (!props.showArcs) return;

    // Clear existing arcs
    arcGroup.clear();
    arcLines = [];

    const arcData = arcs();
    if (!arcData || arcData.length === 0) return;

    arcData.forEach((arc: GlobeArc) => {
      const startPos = latLonToVector3(arc.source.lat, arc.source.lng, GLOBE_RADIUS + 0.5);
      const endPos = latLonToVector3(arc.target.lat, arc.target.lng, GLOBE_RADIUS + 0.5);

      // Create bezier curve for arc
      const midPoint = new THREE.Vector3()
        .addVectors(startPos, endPos)
        .multiplyScalar(0.5)
        .normalize()
        .multiplyScalar(GLOBE_RADIUS + 20 * arc.strength);

      const curve = new THREE.QuadraticBezierCurve3(startPos, midPoint, endPos);
      const points = curve.getPoints(50);
      const geometry = new THREE.BufferGeometry().setFromPoints(points);

      // Use arc color (gradient start)
      const arcColor = new THREE.Color(arc.color[0]);

      const material = new THREE.LineBasicMaterial({
        color: arcColor,
        transparent: true,
        opacity: 0.6 * arc.strength,
        linewidth: 2,
      });

      const line = new THREE.Line(geometry, material);
      arcGroup.add(line);
      arcLines.push(line);
    });
  }

  async function updateBoundaries() {
    if (!filters().showBoundaries) return;

    // Clear existing boundaries
    boundaryGroup.clear();
    boundaryLines = [];

    console.log(`🗺️ Loading world boundaries...`);

    try {
      // Fetch GeoJSON for ALL world countries (110m resolution for performance)
      const response = await fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json');
      const topoData = await response.json();
      
      // Convert TopoJSON to GeoJSON (inline conversion for simplicity)
      // This is a simplified approach - production would use topojson-client library
      console.log(`📥 Loaded TopoJSON data`);
      
      // For now, use alternative GeoJSON source
      const geoResponse = await fetch('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson');
      const geoData = await geoResponse.json();
      
      console.log(`📥 Loaded ${geoData.features?.length || 0} countries from GeoJSON`);
      
      // Create sentiment map from boundary data (ISO2 codes)
      const boundaryData = boundaries();
      const sentimentMap: Record<string, { sentiment: number, count: number }> = {};
      
      if (boundaryData && boundaryData.length > 0) {
        boundaryData.forEach((country: any) => {
          sentimentMap[country.country_code] = {
            sentiment: country.sentiment_score,
            count: country.article_count
          };
        });
        console.log(`📊 Sentiment data available for ${Object.keys(sentimentMap).length} countries`);
      }

      // Comprehensive ISO2 to ISO3 mapping
      const iso2ToIso3: Record<string, string> = {
        'US': 'USA', 'GB': 'GBR', 'JP': 'JPN', 'CN': 'CHN', 'DE': 'DEU',
        'FR': 'FRA', 'IN': 'IND', 'BR': 'BRA', 'CA': 'CAN', 'AU': 'AUS',
        'ZA': 'ZAF', 'NG': 'NGA', 'EG': 'EGY', 'KE': 'KEN', 'SA': 'SAU',
        'AE': 'ARE', 'RU': 'RUS', 'IT': 'ITA', 'ES': 'ESP', 'MX': 'MEX',
        'KR': 'KOR', 'ID': 'IDN', 'TR': 'TUR', 'AR': 'ARG', 'PL': 'POL',
        'TH': 'THA', 'MY': 'MYS', 'SG': 'SGP', 'PH': 'PHL', 'VN': 'VNM',
        'GH': 'GHA', 'DZ': 'DZA', 'MA': 'MAR', 'TN': 'TUN', 'LY': 'LBY',
        'ET': 'ETH', 'TZ': 'TZA', 'UG': 'UGA', 'AO': 'AGO', 'MZ': 'MOZ',
        'BW': 'BWA', 'ZM': 'ZMB', 'ZW': 'ZWE', 'MW': 'MWI', 'CD': 'COD',
        'CI': 'CIV', 'CM': 'CMR', 'SN': 'SEN', 'ML': 'MLI', 'NE': 'NER',
        'TD': 'TCD', 'SD': 'SDN', 'SS': 'SSD', 'ER': 'ERI', 'SO': 'SOM',
      };

      // Reverse map for quick lookup
      const iso3ToIso2: Record<string, string> = {};
      Object.entries(iso2ToIso3).forEach(([iso2, iso3]) => {
        iso3ToIso2[iso3] = iso2;
      });

      let renderedCount = 0;
      let withSentimentCount = 0;

      // Process each country in GeoJSON - RENDER ALL COUNTRIES
      geoData.features?.forEach((feature: any) => {
        const iso3 = feature.properties?.ISO_A3 || feature.properties?.iso_a3 || feature.id;
        if (!iso3 || iso3 === '-99') return; // Skip invalid codes

        const countryName = feature.properties?.NAME || feature.properties?.name || 'Unknown';
        
        // Find matching sentiment data
        const iso2 = iso3ToIso2[iso3];
        const hasSentiment = iso2 && sentimentMap[iso2];
        
        // Determine color: sentiment-based if available, dull grey otherwise
        let color = 0x666666; // Visible dull grey for countries without news
        let opacity = 0.35; // Medium opacity for visibility
        
        if (hasSentiment) {
          const sentiment = sentimentMap[iso2].sentiment;
          if (sentiment > 0.3) {
            color = 0x00ff88; // Bright green for positive
          } else if (sentiment < -0.3) {
            color = 0xff3366; // Bright red for negative
          } else {
            color = 0x4499ff; // Bright blue for neutral
          }
          opacity = 0.6; // Higher opacity for countries with news
          withSentimentCount++;
        }

        // Process geometry (can be Polygon or MultiPolygon)
        const geometries = feature.geometry.type === 'MultiPolygon'
          ? feature.geometry.coordinates
          : [feature.geometry.coordinates];

        geometries.forEach((polygon: any) => {
          // Each polygon is an array of linear rings (first is outer, rest are holes)
          const outerRing = polygon[0];
          
          // Skip if too few points
          if (!outerRing || outerRing.length < 3) return;
          
          // Convert coordinates to 3D points on globe
          const points = outerRing.map((coord: number[]) => {
            const [lng, lat] = coord;
            return latLonToVector3(lat, lng, GLOBE_RADIUS + 0.2);
          });

          // Create line geometry for border
          const geometry = new THREE.BufferGeometry().setFromPoints(points);
          
          const material = new THREE.LineBasicMaterial({
            color: color,
            transparent: true,
            opacity: opacity,
            linewidth: 2,
          });

          const line = new THREE.LineLoop(geometry, material);
          
          // Store country data for click interaction
          line.userData = {
            type: 'country',
            iso3: iso3,
            iso2: iso2 || null,
            name: countryName,
            hasSentiment: hasSentiment,
            sentiment: hasSentiment ? sentimentMap[iso2].sentiment : null
          };
          
          boundaryGroup.add(line);
          boundaryLines.push(line);

          // Add very subtle fill only for countries with sentiment
          if (hasSentiment && outerRing.length < 200) {
            const fillGeometry = new THREE.BufferGeometry().setFromPoints(points);
            const fillMaterial = new THREE.MeshBasicMaterial({
              color: color,
              transparent: true,
              opacity: 0.03,
              side: THREE.DoubleSide,
            });

            const fillMesh = new THREE.Mesh(fillGeometry, fillMaterial);
            fillMesh.userData = line.userData; // Same data for click handling
            boundaryGroup.add(fillMesh);
          }
        });

        renderedCount++;
      });

      console.log(`✅ Rendered ALL ${renderedCount} countries (${withSentimentCount} with news sentiment)`);
      console.log(`   🟢 Green: Positive news | 🔴 Red: Negative news | 🔵 Blue: Neutral | ⚫ Grey: No news`);
    } catch (error) {
      console.error('❌ Error loading country boundaries:', error);
      console.log('⚠️  Political boundaries not available');
    }
  }

  async function fetchCountryDetails(countryCode: string) {
    try {
      console.log(`📡 Fetching details for country: ${countryCode}`);
      
      const response = await fetch(`/api/v1/globe/countries/${countryCode}?timeframe=24h`);
      
      if (!response.ok) {
        if (response.status === 404) {
          console.warn(`⚠️ Country ${countryCode} not found in database`);
          // Show basic info for countries not in database
          setSelectedCountry({
            code: countryCode,
            name: countryCode,
            flag: getCountryFlag(countryCode),
            sentiment: null,
            news_count: 0,
            exchanges_count: 0,
            assets_count: 0,
            gdp: null,
            gdp_growth: null,
            inflation: null,
            unemployment: null,
          });
          return;
        }
        throw new Error(`Failed to fetch country details: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Format data for CountryModal
      const countryData: CountryData = {
        code: data.code,
        name: data.name,
        flag: data.flag,
        gdp: data.gdp,
        gdp_growth: data.gdp_growth,
        inflation: data.inflation,
        unemployment: data.unemployment,
        sentiment: data.sentiment,
        news_count: data.news_count,
        exchanges_count: data.exchanges_count,
        assets_count: data.assets_count,
        top_news: data.top_news,
        recent_news: data.recent_news || [],
      };
      
      setSelectedCountry(countryData);
      console.log(`✅ Country data loaded:`, countryData);
      
    } catch (error) {
      console.error(`❌ Error fetching country details:`, error);
      // Show basic info even if API fails
      setSelectedCountry({
        code: countryCode,
        name: countryCode,
        flag: getCountryFlag(countryCode),
        sentiment: null,
        news_count: 0,
        exchanges_count: 0,
        assets_count: 0,
        gdp: null,
        gdp_growth: null,
        inflation: null,
        unemployment: null,
      });
    }
  }

  function getCountryFlag(countryCode: string): string {
    if (!countryCode || countryCode.length !== 2) return '🌍';
    // Convert to regional indicator symbols
    return String.fromCodePoint(...[...countryCode.toUpperCase()].map(c => 127397 + c.charCodeAt(0)));
  }

  function zoomToLocation(lat: number, lng: number, altitude: number = 80) {
    console.log(`🎯 Zooming to location: ${lat}, ${lng}`);
    
    // Stop auto-rotation
    if (controls) {
      controls.autoRotate = false;
    }
    
    // Calculate target camera position (outside the globe)
    const targetCameraPos = latLonToVector3(lat, lng, GLOBE_RADIUS + altitude);
    
    // Calculate look-at target (on the globe surface)
    const lookAtPos = latLonToVector3(lat, lng, GLOBE_RADIUS);
    
    // Clear any existing tweens
    TWEEN.removeAll();
    
    // Animate camera position
    new TWEEN.Tween(camera.position)
      .to(targetCameraPos, 2000)
      .easing(TWEEN.Easing.Cubic.InOut)
      .start();
    
    // Animate camera target (what it's looking at)
    new TWEEN.Tween(controls.target)
      .to(lookAtPos, 2000)
      .easing(TWEEN.Easing.Cubic.InOut)
      .onUpdate(() => {
        controls.update();
      })
      .start();
  }

  function handleSearchSelect(result: SearchResult) {
    console.log(`🔍 Search selected:`, result);
    zoomToLocation(result.lat, result.lng, 60);
  }

  function latLonToVector3(lat: number, lon: number, radius: number): THREE.Vector3 {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);

    const x = -(radius * Math.sin(phi) * Math.cos(theta));
    const y = radius * Math.cos(phi);
    const z = radius * Math.sin(phi) * Math.sin(theta);

    return new THREE.Vector3(x, y, z);
  }

  function animate() {
    animationId = requestAnimationFrame(animate);

    // Update TWEEN animations
    TWEEN.update();

    // Update controls
    controls.update();

    // Rotate the earth on its axis (if auto-rotate is enabled via props)
    // We use manual rotation here instead of controls.autoRotate to simulate Earth's spin
    if (props.autoRotate !== false) {
      globeGroup.rotation.y += 0.0005;
    }

    // Slow cloud drift for realism
    if (cloudMesh) {
      cloudMesh.rotation.y += 0.0003;
    }

    const time = Date.now() * 0.001;

    // Distance-based marker scaling + event pulsing
    markers.forEach(({ mesh }) => {
      const distance = camera.position.distanceTo(mesh.position);
      const scale = distance / 500;
      const originalSize = mesh.userData.originalSize || MARKER_BASE_SIZE;
      let scaledSize = scale * originalSize;
      
      // Add pulsing effect for active events
      if (mesh.userData.pulseSpeed > 0) {
        const pulse = 1 + Math.sin(time * mesh.userData.pulseSpeed + mesh.userData.pulsePhase) * 0.2;
        scaledSize *= pulse;
        
        // Pulse opacity too
        const material = mesh.material as THREE.MeshBasicMaterial;
        material.opacity = 0.7 + Math.sin(time * mesh.userData.pulseSpeed + mesh.userData.pulsePhase) * 0.2;
      }
      
      mesh.scale.set(scaledSize, scaledSize, scaledSize);
    });
    
    // Scale asset markers similarly
    assetMarkers.forEach(({ mesh }) => {
      const distance = camera.position.distanceTo(mesh.position);
      const scale = distance / 500;
      const originalSize = mesh.userData.originalSize || ASSET_BASE_SIZE;
      let scaledSize = scale * originalSize;
      
      // Add pulsing for active events
      if (mesh.userData.pulseSpeed > 0) {
        const pulse = 1 + Math.sin(time * mesh.userData.pulseSpeed + mesh.userData.pulsePhase) * 0.25;
        scaledSize *= pulse;
        
        const material = mesh.material as THREE.MeshBasicMaterial;
        material.opacity = 0.7 + Math.sin(time * mesh.userData.pulseSpeed + mesh.userData.pulsePhase) * 0.2;
      }
      
      mesh.scale.set(scaledSize, scaledSize, scaledSize);
    });

    // Render
    renderer.render(scene, camera);
  }



  function hideAssetModal() {
    setSelectedAsset(null);

    // Animate camera back smoothly
    TWEEN.removeAll();

    new TWEEN.Tween(camera.position)
      .to(lastCameraPos, 1500)
      .easing(TWEEN.Easing.Cubic.InOut)
      .start();

    new TWEEN.Tween(controls.target)
      .to({ x: 0, y: 0, z: 0 }, 1500)
      .easing(TWEEN.Easing.Cubic.InOut)
      .onUpdate(() => {
        controls.update();
      })
      .onComplete(() => {
        controls.autoRotate = props.autoRotate ?? true;
        controls.update();
      })
      .start();
  }

  function hideCountryModal() {
    setSelectedCountry(null);

    // Animate camera back smoothly
    TWEEN.removeAll();

    new TWEEN.Tween(camera.position)
      .to(lastCameraPos, 1500)
      .easing(TWEEN.Easing.Cubic.InOut)
      .start();

    new TWEEN.Tween(controls.target)
      .to({ x: 0, y: 0, z: 0 }, 1500)
      .easing(TWEEN.Easing.Cubic.InOut)
      .onUpdate(() => {
        controls.update();
      })
      .onComplete(() => {
        controls.autoRotate = props.autoRotate ?? true;
        controls.update();
      })
      .start();
  }

  return (
    <div class="relative w-full h-full flex items-center justify-center bg-[#030014]">
      {/* CSS Animations */}
      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }
        
        @keyframes modalSlideIn {
          from {
            opacity: 0;
            transform: scale(0.95) translateY(20px);
          }
          to {
            opacity: 1;
            transform: scale(1) translateY(0);
          }
        }
        
        /* Custom scrollbar for modal */
        .modal-content::-webkit-scrollbar {
          width: 6px;
        }
        
        .modal-content::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 3px;
        }
        
        .modal-content::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.2);
          border-radius: 3px;
        }
        
        .modal-content::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.3);
        }
      `}</style>
      {/* Loading State */}
      <Show when={loading()}>
        <div class="absolute inset-0 flex items-center justify-center bg-terminal-900/50 backdrop-blur-sm z-50">
          <div class="text-center">
            <div class="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-accent-500 mx-auto mb-4"></div>
            <p class="text-white text-lg">🌍 Loading globe data from API...</p>
            <p class="text-gray-400 text-sm mt-2">Fetching exchanges, arcs, and boundaries</p>
          </div>
        </div>
      </Show>

      {/* Error State */}
      <Show when={error()}>
        <div class="absolute top-4 right-4 bg-red-500/20 border border-red-500 text-red-200 px-4 py-2 rounded-lg z-50">
          <p class="font-semibold">Error loading data</p>
          <p class="text-sm">{error()}</p>
        </div>
      </Show>

      {/* Search Component */}
      <div class="absolute top-4 left-4 z-40 w-80">
        <GlobeSearch 
          data={searchData()}
          onSelect={handleSearchSelect}
        />
      </div>

      {/* Filter Panel */}
      <GlobeFilterPanel
        filters={filters()}
        onFiltersChange={setFilters}
        exchangeCount={exchanges()?.length || 0}
        assetCount={assets()?.length || 0}
        statusCounts={statusSummary()}
      />

      {/* Globe Container */}
      <div
        class="relative w-full h-full max-w-[800px] max-h-[800px]"
        style={{
          'box-shadow': '0 0 60px 30px rgba(76, 29, 149, 0.4), 0 0 100px 60px rgba(29, 122, 222, 0.3)',
          'border-radius': '50%',
        }}
      >
        <div ref={containerRef} class="w-full h-full rounded-full" />
      </div>

      {/* Hover Tooltip */}
      <Show when={hoveredExchange()}>
        {(exchange) => (
          <div
            class="absolute pointer-events-none z-40"
            style={{
              left: '50%',
              top: '20%',
              transform: 'translateX(-50%)',
            }}
          >
            <div class="bg-black/80 backdrop-blur-lg border border-white/10 rounded-lg p-3 min-w-[200px]">
              <div class="flex items-center gap-2 mb-2">
                <span class="text-2xl">{exchange().flag}</span>
                <strong class="text-white">{exchange().name}</strong>
              </div>
              <div class="text-sm space-y-1 text-gray-300">
                <div>
                  Articles: <strong class="text-white">{exchange().news_count}</strong>
                </div>
                <div>
                  Sentiment: <strong 
                    class={
                      exchange().sentiment_score > 0.2 ? 'text-green-400' :
                      exchange().sentiment_score < -0.2 ? 'text-red-400' :
                      'text-gray-400'
                    }
                  >
                    {(exchange().sentiment_score * 100).toFixed(1)}%
                  </strong>
                </div>
                <div class="text-xs text-gray-400 mt-1">
                  Click for details →
                </div>
              </div>
            </div>
          </div>
        )}
      </Show>

      {/* Asset Hover Tooltip */}
      <Show when={hoveredAsset()}>
        {(asset) => (
          <div
            class="absolute pointer-events-none z-40"
            style={{
              left: '50%',
              top: '20%',
              transform: 'translateX(-50%)',
            }}
          >
            <div class="bg-black/80 backdrop-blur-lg border border-white/10 rounded-lg p-3 min-w-[250px]">
              <div class="flex items-center gap-2 mb-2">
                <span class="text-2xl">{asset().flag}</span>
                <div>
                  <strong class="text-white block">{asset().name}</strong>
                  <span class="text-xs text-gray-400">{asset().asset_type.replace('_', ' ').toUpperCase()}</span>
                </div>
              </div>
              <div class="text-sm space-y-1 text-gray-300">
                <div>
                  Status: <strong 
                    class={
                      asset().current_status === 'operational' ? 'text-green-400' :
                      asset().current_status === 'issue' ? 'text-red-400' :
                      'text-gray-400'
                    }
                  >
                    {asset().current_status === 'operational' ? '🟢 Operational' :
                     asset().current_status === 'issue' ? '🔴 Issues' :
                     '⚪ Unknown'}
                  </strong>
                </div>
                <div>
                  Importance: <strong class="text-white">{asset().importance_score}/100</strong>
                </div>
                <div>
                  News: <strong class="text-white">{asset().news_count}</strong>
                </div>
                <div class="text-xs text-gray-400 mt-1">
                  Click for details →
                </div>
              </div>
            </div>
          </div>
        )}
      </Show>



      {/* Asset Detail Modal */}
      <Show when={selectedAsset()}>
        {(asset) => (
          <div
            class="absolute inset-0 flex items-center justify-center z-50 bg-black/50 backdrop-blur-sm"
            style={{
              animation: 'fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
            }}
            onClick={(e) => {
              if (e.target === e.currentTarget) hideAssetModal();
            }}
          >
            <div
              class="modal-content bg-terminal-900/95 backdrop-blur-xl border border-terminal-700 rounded-2xl p-5 w-full max-w-md mx-4 shadow-2xl max-h-[55vh] overflow-y-auto"
              style={{
                animation: 'modalSlideIn 0.5s cubic-bezier(0.16, 1, 0.3, 1)',
              }}
            >
              <button
                onClick={hideAssetModal}
                class="absolute top-4 right-4 text-gray-400 hover:text-white text-2xl leading-none"
              >
                ×
              </button>

              <div class="flex items-center gap-3 mb-4">
                <span class="text-4xl">{asset().flag}</span>
                <div>
                  <h2 class="text-xl font-bold text-white">{asset().name}</h2>
                  <p class="text-gray-400 text-sm">
                    {asset().asset_type.replace('_', ' ').toUpperCase()} • {asset().city}, {asset().country}
                  </p>
                </div>
              </div>

              <div class="grid grid-cols-2 gap-3 mb-4">
                <div class="bg-terminal-800/50 rounded-lg p-3">
                  <p class="text-gray-400 text-sm">Status</p>
                  <p 
                    class={`text-xl font-bold ${
                      asset().current_status === 'operational' ? 'text-green-400' :
                      asset().current_status === 'issue' ? 'text-red-400' :
                      'text-gray-400'
                    }`}
                  >
                    {asset().current_status === 'operational' ? '🟢 Operational' :
                     asset().current_status === 'issue' ? '🔴 Issues' :
                     '⚪ Unknown'}
                  </p>
                </div>
                <div class="bg-terminal-800/50 rounded-lg p-3">
                  <p class="text-gray-400 text-sm">News Articles</p>
                  <p class="text-2xl font-bold text-white">{asset().news_count}</p>
                </div>
                <div class="bg-terminal-800/50 rounded-lg p-3">
                  <p class="text-gray-400 text-sm">Importance</p>
                  <p class="text-2xl font-bold text-accent-400">{asset().importance_score}/100</p>
                </div>
                <div class="bg-terminal-800/50 rounded-lg p-3">
                  <p class="text-gray-400 text-sm">Sentiment</p>
                  <p 
                    class={`text-2xl font-bold ${
                      asset().sentiment_score > 0.2 ? 'text-green-400' :
                      asset().sentiment_score < -0.2 ? 'text-red-400' :
                      'text-gray-400'
                    }`}
                  >
                    {(asset().sentiment_score * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              <Show when={asset().description}>
                <div class="mb-4 p-3 bg-terminal-800/30 rounded-lg">
                  <p class="text-gray-300 text-sm">{asset().description}</p>
                </div>
              </Show>

              <Show when={asset().categories && asset().categories.length > 0}>
                <div class="mb-4">
                  <p class="text-gray-400 text-sm mb-2">Top Categories</p>
                  <div class="flex flex-wrap gap-2">
                    <For each={asset().categories.slice(0, 5)}>
                      {(category) => (
                        <span class="bg-accent-500/20 text-accent-300 px-3 py-1 rounded-full text-sm">
                          {category}
                        </span>
                      )}
                    </For>
                  </div>
                </div>
              </Show>

              <Show when={asset().latest_articles && asset().latest_articles.length > 0}>
                <div class="mb-4">
                  <p class="text-gray-400 text-sm mb-2">Latest Articles</p>
                  <div class="space-y-2">
                    <For each={asset().latest_articles.slice(0, 3)}>
                      {(article) => (
                        <div class="p-2 bg-terminal-800/30 rounded text-xs">
                          <p class="text-white font-medium line-clamp-1">{article.title}</p>
                          <p class="text-gray-400 mt-1 line-clamp-2">{article.summary}</p>
                        </div>
                      )}
                    </For>
                  </div>
                </div>
              </Show>

              <Show when={asset().website}>
                <a
                  href={asset().website}
                  target="_blank"
                  rel="noopener noreferrer"
                  class="block w-full bg-accent-500 hover:bg-accent-600 text-white font-semibold py-3 rounded-lg transition-colors text-center"
                >
                  Visit Website →
                </a>
              </Show>
            </div>
          </div>
        )}
      </Show>

      {/* Country Detail Modal - Exchange Style */}
      <Show when={selectedCountry()}>
        {(country) => (
          <div
            class="absolute inset-0 flex items-center justify-center z-50 bg-black/50 backdrop-blur-sm"
            style={{
              animation: 'fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
            }}
            onClick={(e) => {
              if (e.target === e.currentTarget) hideCountryModal();
            }}
          >
            <div
              class="modal-content bg-terminal-900/95 backdrop-blur-xl border border-terminal-700 rounded-2xl p-5 w-full max-w-md mx-4 shadow-2xl max-h-[55vh] overflow-y-auto"
              style={{
                animation: 'modalSlideIn 0.5s cubic-bezier(0.16, 1, 0.3, 1)',
              }}
            >
              <button
                onClick={hideCountryModal}
                class="absolute top-4 right-4 text-gray-400 hover:text-white text-2xl leading-none"
              >
                ×
              </button>

              <div class="flex items-center gap-3 mb-4">
                <span class="text-4xl">{country().flag}</span>
                <div>
                  <h2 class="text-xl font-bold text-white">{country().name}</h2>
                  <p class="text-gray-400 text-sm">
                    {/* Find capital name */}
                    {COUNTRY_CAPITALS.find(c => c.countryCode === country().code)?.capital || country().code}
                  </p>
                </div>
              </div>

              <div class="grid grid-cols-2 gap-3 mb-4">
                <div class="bg-terminal-800/50 rounded-lg p-3">
                  <p class="text-gray-400 text-sm">GDP</p>
                  <p class="text-lg font-bold text-white">
                    {country().gdp 
                      ? `$${(country().gdp! / 1_000_000_000).toFixed(1)}B`
                      : 'N/A'}
                  </p>
                </div>
                <div class="bg-terminal-800/50 rounded-lg p-3">
                  <p class="text-gray-400 text-sm">Inflation</p>
                  <p class={`text-lg font-bold ${(country().inflation ?? 0) > 5 ? 'text-red-400' : 'text-white'}`}>
                    {country().inflation !== null && country().inflation !== undefined
                      ? `${country().inflation!.toFixed(1)}%`
                      : 'N/A'}
                  </p>
                </div>
                <div class="bg-terminal-800/50 rounded-lg p-3">
                  <p class="text-gray-400 text-sm">Exchanges</p>
                  <p class="text-lg font-bold text-white">{country().exchanges_count || 0}</p>
                </div>
                <div class="bg-terminal-800/50 rounded-lg p-3">
                  <p class="text-gray-400 text-sm">Assets</p>
                  <p class="text-lg font-bold text-white">{country().assets_count || 0}</p>
                </div>
              </div>

              {/* Top News */}
              <Show when={country().top_news}>
                <div class="bg-accent-500/10 border border-accent-500/30 rounded-lg p-3 mb-3">
                  <p class="text-accent-400 text-xs mb-2">🔥 Top News</p>
                  <p class="text-white text-sm font-medium mb-1">{country().top_news!.title}</p>
                  <div class="flex items-center justify-between text-xs text-gray-500">
                    <span>{country().top_news!.source}</span>
                    <span class={
                      country().top_news!.sentiment > 0.3 ? 'text-green-400' :
                      country().top_news!.sentiment < -0.3 ? 'text-red-400' :
                      'text-gray-400'
                    }>
                      {(country().top_news!.sentiment * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </Show>

              {/* More News Button */}
              <Show when={country().news_count > 0}>
                <button 
                  onClick={() => {
                    // Navigate to news page with country filter
                    window.location.href = `/news?country=${country().code}`;
                  }}
                  class="w-full bg-accent-600 hover:bg-accent-700 text-white rounded-lg py-2.5 px-4 text-sm font-medium transition-colors"
                >
                  View {country().news_count} More News Articles
                </button>
              </Show>

              <Show when={!country().news_count || country().news_count === 0}>
                <p class="text-center text-gray-500 text-sm py-2">No news available</p>
              </Show>
            </div>
          </div>
        )}
      </Show>

      {/* Asset Detail Modal - Bloomberg Style */}
      <Show when={selectedAssetMarker()}>
        {(asset) => (
          <AssetDetailModal 
            asset={asset()}
            onClose={() => setSelectedAssetMarker(null)}
          />
        )}
      </Show>
    </div>
  );
}
