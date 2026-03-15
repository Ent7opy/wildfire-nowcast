import { useMemo } from "react";
import { Box, MenuItem, Select } from "@mui/material";
import ClearIcon from "@mui/icons-material/Clear";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import PublicIcon from "@mui/icons-material/Public";

import type { FireEvent } from "../types/api";
import {
  ALL_CONTINENTS,
  getContinentForCountry,
  getContinentFromEvent,
  getAdmin1FromEvent,
  getCountryFromEvent,
  type RegionFilterValue,
} from "../utils/continents";

interface Props {
  /** All visible (unfiltered) events — used to derive available dropdown options. */
  events: FireEvent[];
  value: RegionFilterValue;
  onChange: (value: RegionFilterValue) => void;
}

const MENU_PROPS = {
  PaperProps: {
    sx: {
      bgcolor: "#161b22",
      border: "1px solid rgba(255,255,255,0.1)",
      borderRadius: 1.5,
      mt: 0.5,
      boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
      "& .MuiList-root": { py: 0.5 },
      "& .MuiMenuItem-root": {
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: "0.04em",
        color: "#d1d5db",
        py: 0.75,
        px: 1.5,
        minHeight: 0,
        "&:hover": { bgcolor: "rgba(249,115,22,0.1)", color: "#f97316" },
        "&.Mui-selected": {
          bgcolor: "rgba(249,115,22,0.14)",
          color: "#f97316",
          "&:hover": { bgcolor: "rgba(249,115,22,0.2)" },
        },
      },
    },
  },
};

function selectSx(isActive: boolean, isDisabled: boolean) {
  return {
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: "0.07em",
    color: isDisabled ? "#2d3748" : isActive ? "#f97316" : "#6b7280",
    transition: "color 0.15s",
    "& .MuiSelect-select": {
      py: "2px !important",
      px: "2px !important",
      pr: "18px !important",
      minWidth: isDisabled ? 52 : 72,
      maxWidth: 148,
      overflow: "hidden",
      textOverflow: "ellipsis",
      whiteSpace: "nowrap",
    },
    "& .MuiSelect-icon": {
      color: isDisabled ? "#1f2937" : isActive ? "#f97316" : "#374151",
      fontSize: 14,
      right: 0,
      top: "calc(50% - 7px)",
      transition: "color 0.15s",
    },
    "& .MuiInput-root:before, & .MuiInput-root:after, & .MuiInput-root:hover:before": {
      borderBottom: "none !important",
    },
  } as const;
}

export default function RegionFilter({ events, value, onChange }: Props): JSX.Element {
  const hasSelection = value.continent !== null || value.country !== null || value.admin1 !== null;

  // ── Available options derived from current visible events ──────────────────

  const availableContinents = useMemo(() => {
    const seen = new Set<string>();
    for (const event of events) {
      const continent = getContinentFromEvent(event);
      if (continent) seen.add(continent);
    }
    return ALL_CONTINENTS.filter((c) => seen.has(c));
  }, [events]);

  const availableCountries = useMemo(() => {
    const seen = new Set<string>();
    for (const event of events) {
      const country = getCountryFromEvent(event);
      if (!country) continue;
      if (value.continent) {
        if (getContinentForCountry(country) !== value.continent) continue;
      }
      seen.add(country);
    }
    return Array.from(seen).sort();
  }, [events, value.continent]);

  const availableAdmin1s = useMemo(() => {
    const seen = new Set<string>();
    for (const event of events) {
      const admin1 = getAdmin1FromEvent(event);
      if (!admin1) continue;
      if (value.country) {
        const country = getCountryFromEvent(event);
        if (country?.toLowerCase() !== value.country.toLowerCase()) continue;
      } else if (value.continent) {
        const country = getCountryFromEvent(event);
        if (!country || getContinentForCountry(country) !== value.continent) continue;
      }
      seen.add(admin1);
    }
    return Array.from(seen).sort();
  }, [events, value.continent, value.country]);

  // ── Handlers ───────────────────────────────────────────────────────────────

  const handleContinent = (continent: string) => {
    onChange({ continent: continent || null, country: null, admin1: null });
  };

  const handleCountry = (country: string) => {
    if (!country) {
      onChange({ ...value, country: null, admin1: null });
    } else {
      // Auto-resolve continent when country is chosen directly
      const continent = getContinentForCountry(country) ?? value.continent;
      onChange({ continent, country, admin1: null });
    }
  };

  const handleAdmin1 = (admin1: string) => {
    onChange({ ...value, admin1: admin1 || null });
  };

  const handleClear = () => onChange({ continent: null, country: null, admin1: null });

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        bgcolor: "#0d1117",
        border: hasSelection
          ? "1px solid rgba(249,115,22,0.35)"
          : "1px solid rgba(255,255,255,0.08)",
        borderRadius: 2,
        px: 1,
        py: "3px",
        gap: 0.25,
        transition: "border-color 0.2s",
        minWidth: 0,
      }}
    >
      {/* Globe icon */}
      <PublicIcon
        sx={{
          fontSize: 13,
          color: hasSelection ? "#f97316" : "#374151",
          mr: 0.25,
          flexShrink: 0,
          transition: "color 0.2s",
        }}
      />

      {/* Continent */}
      <Select
        variant="standard"
        value={value.continent ?? ""}
        onChange={(e) => handleContinent(e.target.value)}
        displayEmpty
        disableUnderline
        sx={selectSx(!!value.continent, false)}
        renderValue={(v) => (v ? String(v) : "CONTINENT")}
        MenuProps={MENU_PROPS}
      >
        <MenuItem value="">
          <Box component="span" sx={{ fontStyle: "italic", color: "#6b7280", fontSize: 11 }}>
            Any continent
          </Box>
        </MenuItem>
        {availableContinents.map((c) => (
          <MenuItem key={c} value={c}>
            {c}
          </MenuItem>
        ))}
      </Select>

      {/* Separator */}
      <ChevronRightIcon sx={{ fontSize: 11, color: "#1f2937", flexShrink: 0 }} />

      {/* Country */}
      <Select
        variant="standard"
        value={value.country ?? ""}
        onChange={(e) => handleCountry(e.target.value)}
        displayEmpty
        disableUnderline
        disabled={availableCountries.length === 0}
        sx={selectSx(!!value.country, availableCountries.length === 0)}
        renderValue={(v) => (v ? String(v) : "COUNTRY")}
        MenuProps={MENU_PROPS}
      >
        <MenuItem value="">
          <Box component="span" sx={{ fontStyle: "italic", color: "#6b7280", fontSize: 11 }}>
            Any country
          </Box>
        </MenuItem>
        {availableCountries.map((c) => (
          <MenuItem key={c} value={c}>
            {c}
          </MenuItem>
        ))}
      </Select>

      {/* Separator */}
      <ChevronRightIcon sx={{ fontSize: 11, color: "#1f2937", flexShrink: 0 }} />

      {/* State / Province */}
      <Select
        variant="standard"
        value={value.admin1 ?? ""}
        onChange={(e) => handleAdmin1(e.target.value)}
        displayEmpty
        disableUnderline
        disabled={availableAdmin1s.length === 0}
        sx={selectSx(!!value.admin1, availableAdmin1s.length === 0)}
        renderValue={(v) => (v ? String(v) : "STATE")}
        MenuProps={MENU_PROPS}
      >
        <MenuItem value="">
          <Box component="span" sx={{ fontStyle: "italic", color: "#6b7280", fontSize: 11 }}>
            Any state
          </Box>
        </MenuItem>
        {availableAdmin1s.map((a) => (
          <MenuItem key={a} value={a}>
            {a}
          </MenuItem>
        ))}
      </Select>

      {/* Clear */}
      {hasSelection && (
        <Box
          component="button"
          onClick={handleClear}
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            ml: 0.5,
            p: "2px",
            borderRadius: 0.75,
            border: "none",
            bgcolor: "transparent",
            color: "#4b5563",
            cursor: "pointer",
            flexShrink: 0,
            transition: "color 0.15s",
            "&:hover": { color: "#ef4444" },
          }}
        >
          <ClearIcon sx={{ fontSize: 12 }} />
        </Box>
      )}
    </Box>
  );
}
