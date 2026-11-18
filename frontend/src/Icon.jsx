// Small stroke icons (24px grid, drawn with currentColor) so the UI does not
// depend on emoji, which render differently on every OS.
const PATHS = {
  ask:      <><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /><path d="M11 8v6M8 11h6" /></>,
  upload:   <><path d="M12 15V4" /><path d="m7 9 5-5 5 5" /><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></>,
  graph:    <><circle cx="6" cy="6" r="2.5" /><circle cx="18" cy="8" r="2.5" /><circle cx="9" cy="18" r="2.5" /><path d="M8.3 7.1 15.6 7.7M7 8.4l1.4 7.2M16.6 10l-5.8 6.3" /></>,
  chart:    <><path d="M4 20V4" /><path d="M4 20h16" /><path d="M8 16v-4M12 16V8M16 16v-6" /></>,
  file:     <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /></>,
  send:     <><path d="M12 19V5" /><path d="m5 12 7-7 7 7" /></>,
  copy:     <><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1" /></>,
  check:    <path d="m5 12 5 5 9-10" />,
  x:        <path d="M6 6l12 12M18 6 6 18" />,
  refresh:  <><path d="M20 11a8 8 0 0 0-14.6-4.5L4 8" /><path d="M4 4v4h4" /><path d="M4 13a8 8 0 0 0 14.6 4.5L20 16" /><path d="M20 20v-4h-4" /></>,
  trash:    <><path d="M4 7h16" /><path d="M10 11v6M14 11v6" /><path d="M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12" /><path d="M9 7V4h6v3" /></>,
  plus:     <path d="M12 5v14M5 12h14" />,
  minus:    <path d="M5 12h14" />,
  fit:      <><path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4M20 15v4a1 1 0 0 1-1 1h-4M9 20H5a1 1 0 0 1-1-1v-4" /></>,
  sun:      <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
  moon:     <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />,
  alert:    <><path d="M10.3 4 2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 4a2 2 0 0 0-3.4 0z" /><path d="M12 9v4M12 17h.01" /></>,
  sparkle:  <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" />,
  chevron:  <path d="m6 9 6 6 6-6" />,
  link:     <><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1" /><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" /></>,
  database: <><ellipse cx="12" cy="5.5" rx="7" ry="2.5" /><path d="M5 5.5v13c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-13" /><path d="M5 12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5" /></>,
  layers:   <><path d="m12 3 9 5-9 5-9-5z" /><path d="m3 13 9 5 9-5" /></>,
  code:     <path d="m8 7-5 5 5 5M16 7l5 5-5 5" />,
  route:    <><circle cx="6" cy="18" r="2.5" /><circle cx="18" cy="6" r="2.5" /><path d="M8.5 18H15a3 3 0 0 0 0-6H9a3 3 0 0 1 0-6h6.5" /></>,
  clock:    <><circle cx="12" cy="12" r="8" /><path d="M12 8v4l3 2" /></>,
  filter:   <path d="M4 5h16l-6 8v5l-4 2v-7z" />,
  pill:     <><rect x="3" y="9" width="18" height="6" rx="3" transform="rotate(-45 12 12)" /><path d="m9.5 9.5 5 5" /></>,
  heart:    <path d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z" />,
  flask:    <><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3l-5-9V3" /><path d="M7.5 15h9" /></>,
  kidney:   <><path d="M8 4c-3 0-5 3-5 7s2 9 6 9c2 0 3-1.5 3-3.5S10 13 10 11s2-3 2-4.5S10.5 4 8 4z" /><path d="M16 4c3 0 5 3 5 7s-2 9-6 9" /></>,
}

export function Icon({ name, size = 18, className = '', strokeWidth = 1.75, ...rest }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true" focusable="false" className={`icon ${className}`} {...rest}>
      {PATHS[name]}
    </svg>
  )
}

// Brand mark: three linked nodes inside a rounded square
export function Logo({ size = 30 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" focusable="false">
      <rect width="32" height="32" rx="9" fill="var(--accent)" />
      <path d="M10 11.5 21.5 10M10.5 12.5l4.5 9M21 11.5l-5 9.5" stroke="var(--on-accent)"
        strokeWidth="1.6" strokeLinecap="round" opacity="0.75" />
      <circle cx="10" cy="11.5" r="3" fill="var(--on-accent)" />
      <circle cx="22" cy="10" r="2.4" fill="var(--on-accent)" />
      <circle cx="15.5" cy="22" r="3.4" fill="var(--on-accent)" />
    </svg>
  )
}
