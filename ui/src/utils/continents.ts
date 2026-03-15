import type { FireEvent } from "../types/api";

export interface RegionFilterValue {
  continent: string | null;
  country: string | null;
  admin1: string | null;
}

export const EMPTY_REGION_FILTER: RegionFilterValue = {
  continent: null,
  country: null,
  admin1: null,
};

/**
 * Canonical continent order for display.
 */
export const ALL_CONTINENTS = [
  "North America",
  "South America",
  "Europe",
  "Africa",
  "Asia",
  "Oceania",
] as const;

/**
 * Approximate map viewports for each continent — used for fly-to when no events
 * are yet visible in the selected region.
 */
export const CONTINENT_VIEWPORTS: Record<string, { lat: number; lon: number; zoom: number }> = {
  "North America": { lat: 45, lon: -100, zoom: 3 },
  "South America": { lat: -15, lon: -60, zoom: 3 },
  "Europe": { lat: 50, lon: 10, zoom: 4 },
  "Africa": { lat: 2, lon: 20, zoom: 3 },
  "Asia": { lat: 35, lon: 100, zoom: 3 },
  "Oceania": { lat: -27, lon: 134, zoom: 4 },
};

/**
 * Country name (lowercase) → continent. Covers all major wildfire-prone nations
 * plus a wide range of others. Country names come from Nominatim reverse geocoding.
 */
const COUNTRY_CONTINENT: Record<string, string> = {
  // ── North America ──────────────────────────────────────────────────────────
  "united states": "North America",
  "usa": "North America",
  "united states of america": "North America",
  "canada": "North America",
  "mexico": "North America",
  "guatemala": "North America",
  "belize": "North America",
  "honduras": "North America",
  "el salvador": "North America",
  "nicaragua": "North America",
  "costa rica": "North America",
  "panama": "North America",
  "cuba": "North America",
  "haiti": "North America",
  "dominican republic": "North America",
  "jamaica": "North America",
  "puerto rico": "North America",
  "trinidad and tobago": "North America",
  "barbados": "North America",
  "bahamas": "North America",
  "the bahamas": "North America",
  "grenada": "North America",
  "saint lucia": "North America",
  "saint vincent and the grenadines": "North America",
  "antigua and barbuda": "North America",
  "dominica": "North America",
  "saint kitts and nevis": "North America",

  // ── South America ──────────────────────────────────────────────────────────
  "brazil": "South America",
  "argentina": "South America",
  "bolivia": "South America",
  "chile": "South America",
  "colombia": "South America",
  "ecuador": "South America",
  "peru": "South America",
  "venezuela": "South America",
  "paraguay": "South America",
  "uruguay": "South America",
  "guyana": "South America",
  "suriname": "South America",
  "french guiana": "South America",
  "trinidad": "South America",

  // ── Europe ─────────────────────────────────────────────────────────────────
  "russia": "Europe",
  "russian federation": "Europe",
  "france": "Europe",
  "spain": "Europe",
  "portugal": "Europe",
  "italy": "Europe",
  "greece": "Europe",
  "turkey": "Europe",
  "türkiye": "Europe",
  "germany": "Europe",
  "sweden": "Europe",
  "norway": "Europe",
  "finland": "Europe",
  "ukraine": "Europe",
  "united kingdom": "Europe",
  "uk": "Europe",
  "great britain": "Europe",
  "england": "Europe",
  "scotland": "Europe",
  "wales": "Europe",
  "northern ireland": "Europe",
  "poland": "Europe",
  "romania": "Europe",
  "bulgaria": "Europe",
  "croatia": "Europe",
  "serbia": "Europe",
  "albania": "Europe",
  "north macedonia": "Europe",
  "bosnia and herzegovina": "Europe",
  "montenegro": "Europe",
  "hungary": "Europe",
  "czech republic": "Europe",
  "czechia": "Europe",
  "slovakia": "Europe",
  "austria": "Europe",
  "switzerland": "Europe",
  "netherlands": "Europe",
  "belgium": "Europe",
  "denmark": "Europe",
  "iceland": "Europe",
  "ireland": "Europe",
  "latvia": "Europe",
  "lithuania": "Europe",
  "estonia": "Europe",
  "belarus": "Europe",
  "moldova": "Europe",
  "republic of moldova": "Europe",
  "slovenia": "Europe",
  "cyprus": "Europe",
  "malta": "Europe",
  "kosovo": "Europe",
  "luxembourg": "Europe",
  "liechtenstein": "Europe",
  "monaco": "Europe",
  "san marino": "Europe",
  "andorra": "Europe",
  "georgia": "Europe",
  "armenia": "Europe",
  "azerbaijan": "Europe",
  "north cyprus": "Europe",

  // ── Africa ─────────────────────────────────────────────────────────────────
  "south africa": "Africa",
  "nigeria": "Africa",
  "democratic republic of the congo": "Africa",
  "dr congo": "Africa",
  "congo": "Africa",
  "republic of the congo": "Africa",
  "tanzania": "Africa",
  "united republic of tanzania": "Africa",
  "kenya": "Africa",
  "ethiopia": "Africa",
  "angola": "Africa",
  "zimbabwe": "Africa",
  "mozambique": "Africa",
  "madagascar": "Africa",
  "cameroon": "Africa",
  "zambia": "Africa",
  "malawi": "Africa",
  "ghana": "Africa",
  "ivory coast": "Africa",
  "côte d'ivoire": "Africa",
  "cote d'ivoire": "Africa",
  "senegal": "Africa",
  "mali": "Africa",
  "burkina faso": "Africa",
  "niger": "Africa",
  "chad": "Africa",
  "sudan": "Africa",
  "south sudan": "Africa",
  "somalia": "Africa",
  "eritrea": "Africa",
  "djibouti": "Africa",
  "rwanda": "Africa",
  "burundi": "Africa",
  "uganda": "Africa",
  "central african republic": "Africa",
  "gabon": "Africa",
  "equatorial guinea": "Africa",
  "namibia": "Africa",
  "botswana": "Africa",
  "lesotho": "Africa",
  "eswatini": "Africa",
  "swaziland": "Africa",
  "sierra leone": "Africa",
  "liberia": "Africa",
  "guinea": "Africa",
  "guinea-bissau": "Africa",
  "gambia": "Africa",
  "the gambia": "Africa",
  "mauritania": "Africa",
  "algeria": "Africa",
  "morocco": "Africa",
  "tunisia": "Africa",
  "libya": "Africa",
  "egypt": "Africa",
  "benin": "Africa",
  "togo": "Africa",
  "cape verde": "Africa",
  "cabo verde": "Africa",
  "sao tome and principe": "Africa",
  "comoros": "Africa",
  "seychelles": "Africa",
  "mauritius": "Africa",
  "réunion": "Africa",
  "reunion": "Africa",
  "mayotte": "Africa",

  // ── Asia ───────────────────────────────────────────────────────────────────
  "china": "Asia",
  "people's republic of china": "Asia",
  "india": "Asia",
  "indonesia": "Asia",
  "philippines": "Asia",
  "vietnam": "Asia",
  "viet nam": "Asia",
  "thailand": "Asia",
  "malaysia": "Asia",
  "myanmar": "Asia",
  "burma": "Asia",
  "cambodia": "Asia",
  "laos": "Asia",
  "lao people's democratic republic": "Asia",
  "japan": "Asia",
  "south korea": "Asia",
  "republic of korea": "Asia",
  "north korea": "Asia",
  "taiwan": "Asia",
  "mongolia": "Asia",
  "kazakhstan": "Asia",
  "uzbekistan": "Asia",
  "kyrgyzstan": "Asia",
  "tajikistan": "Asia",
  "turkmenistan": "Asia",
  "afghanistan": "Asia",
  "pakistan": "Asia",
  "bangladesh": "Asia",
  "sri lanka": "Asia",
  "nepal": "Asia",
  "bhutan": "Asia",
  "iran": "Asia",
  "islamic republic of iran": "Asia",
  "iraq": "Asia",
  "syria": "Asia",
  "syrian arab republic": "Asia",
  "israel": "Asia",
  "jordan": "Asia",
  "lebanon": "Asia",
  "saudi arabia": "Asia",
  "yemen": "Asia",
  "oman": "Asia",
  "united arab emirates": "Asia",
  "uae": "Asia",
  "qatar": "Asia",
  "bahrain": "Asia",
  "kuwait": "Asia",
  "brunei": "Asia",
  "singapore": "Asia",
  "timor-leste": "Asia",
  "east timor": "Asia",
  "maldives": "Asia",

  // ── Oceania ────────────────────────────────────────────────────────────────
  "australia": "Oceania",
  "new zealand": "Oceania",
  "fiji": "Oceania",
  "papua new guinea": "Oceania",
  "solomon islands": "Oceania",
  "vanuatu": "Oceania",
  "samoa": "Oceania",
  "tonga": "Oceania",
  "kiribati": "Oceania",
  "micronesia": "Oceania",
  "federated states of micronesia": "Oceania",
  "palau": "Oceania",
  "marshall islands": "Oceania",
  "nauru": "Oceania",
  "tuvalu": "Oceania",
  "new caledonia": "Oceania",
  "french polynesia": "Oceania",
};

export function getContinentForCountry(country: string): string | null {
  return COUNTRY_CONTINENT[country.toLowerCase()] ?? null;
}

export function getCountryFromEvent(event: FireEvent): string | null {
  const v = event.admin0_name?.trim() || event.country?.trim();
  return v && v.length > 0 ? v : null;
}

export function getAdmin1FromEvent(event: FireEvent): string | null {
  const v = event.admin1_name?.trim();
  return v && v.length > 0 ? v : null;
}

export function getContinentFromEvent(event: FireEvent): string | null {
  const country = getCountryFromEvent(event);
  return country ? getContinentForCountry(country) : null;
}

/**
 * Returns true if the event matches the given region filter.
 * An empty filter (all nulls) matches everything.
 */
export function matchesRegionFilter(event: FireEvent, filter: RegionFilterValue): boolean {
  if (!filter.continent && !filter.country && !filter.admin1) return true;

  const country = getCountryFromEvent(event);
  const admin1 = getAdmin1FromEvent(event);

  if (filter.admin1) {
    return admin1?.toLowerCase() === filter.admin1.toLowerCase();
  }

  if (filter.country) {
    return country?.toLowerCase() === filter.country.toLowerCase();
  }

  if (filter.continent) {
    const eventContinent = country ? getContinentForCountry(country) : null;
    return eventContinent === filter.continent;
  }

  return true;
}

/**
 * Format a region filter value as a human-readable string for assistant context.
 * Returns null when no filter is active.
 */
export function formatRegionFilter(filter: RegionFilterValue): string | null {
  const parts = [filter.admin1, filter.country, filter.continent].filter(Boolean);
  return parts.length > 0 ? parts.join(", ") : null;
}
