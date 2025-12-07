/**
 * Country Capitals Data
 * Contains capital city coordinates for all countries
 */

export interface CapitalData {
  country: string;
  countryCode: string;
  capital: string;
  lat: number;
  lng: number;
  flag: string;
}

export const COUNTRY_CAPITALS: CapitalData[] = [
  // Africa
  { country: "Nigeria", countryCode: "NG", capital: "Abuja", lat: 9.0765, lng: 7.3986, flag: "🇳🇬" },
  { country: "South Africa", countryCode: "ZA", capital: "Pretoria", lat: -25.7461, lng: 28.1881, flag: "🇿🇦" },
  { country: "Kenya", countryCode: "KE", capital: "Nairobi", lat: -1.2864, lng: 36.8172, flag: "🇰🇪" },
  { country: "Egypt", countryCode: "EG", capital: "Cairo", lat: 30.0444, lng: 31.2357, flag: "🇪🇬" },
  { country: "Ethiopia", countryCode: "ET", capital: "Addis Ababa", lat: 9.0320, lng: 38.7469, flag: "🇪🇹" },
  { country: "Ghana", countryCode: "GH", capital: "Accra", lat: 5.6037, lng: -0.1870, flag: "🇬🇭" },
  { country: "Morocco", countryCode: "MA", capital: "Rabat", lat: 34.0209, lng: -6.8416, flag: "🇲🇦" },
  { country: "Tanzania", countryCode: "TZ", capital: "Dodoma", lat: -6.1630, lng: 35.7516, flag: "🇹🇿" },
  { country: "Algeria", countryCode: "DZ", capital: "Algiers", lat: 36.7372, lng: 3.0865, flag: "🇩🇿" },
  { country: "Uganda", countryCode: "UG", capital: "Kampala", lat: 0.3476, lng: 32.5825, flag: "🇺🇬" },
  
  // Americas
  { country: "United States", countryCode: "US", capital: "Washington D.C.", lat: 38.9072, lng: -77.0369, flag: "🇺🇸" },
  { country: "Canada", countryCode: "CA", capital: "Ottawa", lat: 45.4215, lng: -75.6972, flag: "🇨🇦" },
  { country: "Brazil", countryCode: "BR", capital: "Brasília", lat: -15.7975, lng: -47.8919, flag: "🇧🇷" },
  { country: "Mexico", countryCode: "MX", capital: "Mexico City", lat: 19.4326, lng: -99.1332, flag: "🇲🇽" },
  { country: "Argentina", countryCode: "AR", capital: "Buenos Aires", lat: -34.6037, lng: -58.3816, flag: "🇦🇷" },
  { country: "Colombia", countryCode: "CO", capital: "Bogotá", lat: 4.7110, lng: -74.0721, flag: "🇨🇴" },
  { country: "Chile", countryCode: "CL", capital: "Santiago", lat: -33.4489, lng: -70.6693, flag: "🇨🇱" },
  { country: "Peru", countryCode: "PE", capital: "Lima", lat: -12.0464, lng: -77.0428, flag: "🇵🇪" },
  
  // Asia
  { country: "China", countryCode: "CN", capital: "Beijing", lat: 39.9042, lng: 116.4074, flag: "🇨🇳" },
  { country: "Japan", countryCode: "JP", capital: "Tokyo", lat: 35.6762, lng: 139.6503, flag: "🇯🇵" },
  { country: "India", countryCode: "IN", capital: "New Delhi", lat: 28.6139, lng: 77.2090, flag: "🇮🇳" },
  { country: "South Korea", countryCode: "KR", capital: "Seoul", lat: 37.5665, lng: 126.9780, flag: "🇰🇷" },
  { country: "Indonesia", countryCode: "ID", capital: "Jakarta", lat: -6.2088, lng: 106.8456, flag: "🇮🇩" },
  { country: "Thailand", countryCode: "TH", capital: "Bangkok", lat: 13.7563, lng: 100.5018, flag: "🇹🇭" },
  { country: "Vietnam", countryCode: "VN", capital: "Hanoi", lat: 21.0285, lng: 105.8542, flag: "🇻🇳" },
  { country: "Philippines", countryCode: "PH", capital: "Manila", lat: 14.5995, lng: 120.9842, flag: "🇵🇭" },
  { country: "Malaysia", countryCode: "MY", capital: "Kuala Lumpur", lat: 3.1390, lng: 101.6869, flag: "🇲🇾" },
  { country: "Singapore", countryCode: "SG", capital: "Singapore", lat: 1.3521, lng: 103.8198, flag: "🇸🇬" },
  { country: "Pakistan", countryCode: "PK", capital: "Islamabad", lat: 33.6844, lng: 73.0479, flag: "🇵🇰" },
  { country: "Bangladesh", countryCode: "BD", capital: "Dhaka", lat: 23.8103, lng: 90.4125, flag: "🇧🇩" },
  { country: "Saudi Arabia", countryCode: "SA", capital: "Riyadh", lat: 24.7136, lng: 46.6753, flag: "🇸🇦" },
  { country: "UAE", countryCode: "AE", capital: "Abu Dhabi", lat: 24.4539, lng: 54.3773, flag: "🇦🇪" },
  { country: "Turkey", countryCode: "TR", capital: "Ankara", lat: 39.9334, lng: 32.8597, flag: "🇹🇷" },
  { country: "Israel", countryCode: "IL", capital: "Jerusalem", lat: 31.7683, lng: 35.2137, flag: "🇮🇱" },
  { country: "Iran", countryCode: "IR", capital: "Tehran", lat: 35.6892, lng: 51.3890, flag: "🇮🇷" },
  
  // Europe
  { country: "United Kingdom", countryCode: "GB", capital: "London", lat: 51.5074, lng: -0.1278, flag: "🇬🇧" },
  { country: "Germany", countryCode: "DE", capital: "Berlin", lat: 52.5200, lng: 13.4050, flag: "🇩🇪" },
  { country: "France", countryCode: "FR", capital: "Paris", lat: 48.8566, lng: 2.3522, flag: "🇫🇷" },
  { country: "Italy", countryCode: "IT", capital: "Rome", lat: 41.9028, lng: 12.4964, flag: "🇮🇹" },
  { country: "Spain", countryCode: "ES", capital: "Madrid", lat: 40.4168, lng: -3.7038, flag: "🇪🇸" },
  { country: "Russia", countryCode: "RU", capital: "Moscow", lat: 55.7558, lng: 37.6173, flag: "🇷🇺" },
  { country: "Poland", countryCode: "PL", capital: "Warsaw", lat: 52.2297, lng: 21.0122, flag: "🇵🇱" },
  { country: "Netherlands", countryCode: "NL", capital: "Amsterdam", lat: 52.3676, lng: 4.9041, flag: "🇳🇱" },
  { country: "Belgium", countryCode: "BE", capital: "Brussels", lat: 50.8503, lng: 4.3517, flag: "🇧🇪" },
  { country: "Sweden", countryCode: "SE", capital: "Stockholm", lat: 59.3293, lng: 18.0686, flag: "🇸🇪" },
  { country: "Norway", countryCode: "NO", capital: "Oslo", lat: 59.9139, lng: 10.7522, flag: "🇳🇴" },
  { country: "Denmark", countryCode: "DK", capital: "Copenhagen", lat: 55.6761, lng: 12.5683, flag: "🇩🇰" },
  { country: "Finland", countryCode: "FI", capital: "Helsinki", lat: 60.1699, lng: 24.9384, flag: "🇫🇮" },
  { country: "Austria", countryCode: "AT", capital: "Vienna", lat: 48.2082, lng: 16.3738, flag: "🇦🇹" },
  { country: "Switzerland", countryCode: "CH", capital: "Bern", lat: 46.9481, lng: 7.4474, flag: "🇨🇭" },
  { country: "Portugal", countryCode: "PT", capital: "Lisbon", lat: 38.7223, lng: -9.1393, flag: "🇵🇹" },
  { country: "Greece", countryCode: "GR", capital: "Athens", lat: 37.9838, lng: 23.7275, flag: "🇬🇷" },
  { country: "Czech Republic", countryCode: "CZ", capital: "Prague", lat: 50.0755, lng: 14.4378, flag: "🇨🇿" },
  { country: "Romania", countryCode: "RO", capital: "Bucharest", lat: 44.4268, lng: 26.1025, flag: "🇷🇴" },
  { country: "Hungary", countryCode: "HU", capital: "Budapest", lat: 47.4979, lng: 19.0402, flag: "🇭🇺" },
  { country: "Ireland", countryCode: "IE", capital: "Dublin", lat: 53.3498, lng: -6.2603, flag: "🇮🇪" },
  { country: "Ukraine", countryCode: "UA", capital: "Kyiv", lat: 50.4501, lng: 30.5234, flag: "🇺🇦" },
  
  // Oceania
  { country: "Australia", countryCode: "AU", capital: "Canberra", lat: -35.2809, lng: 149.1300, flag: "🇦🇺" },
  { country: "New Zealand", countryCode: "NZ", capital: "Wellington", lat: -41.2865, lng: 174.7762, flag: "🇳🇿" },
];
