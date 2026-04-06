"use client";

const PALETTE = [
  { bg: "#0C4A6E", text: "#BAE6FD" }, // sky
  { bg: "#164E63", text: "#A5F3FC" }, // cyan
  { bg: "#134E4A", text: "#99F6E4" }, // teal
  { bg: "#14532D", text: "#BBF7D0" }, // green
  { bg: "#3B0764", text: "#E9D5FF" }, // purple
  { bg: "#4C1D95", text: "#C4B5FD" }, // violet
  { bg: "#701A75", text: "#F0ABFC" }, // fuchsia
  { bg: "#7C2D12", text: "#FED7AA" }, // orange
  { bg: "#78350F", text: "#FDE68A" }, // amber
  { bg: "#1E3A5F", text: "#93C5FD" }, // blue
];

function hashCode(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

function getInitials(firstName?: string | null, middleName?: string | null): string {
  const first = firstName?.trim()?.[0]?.toUpperCase() ?? "";
  const second = middleName?.trim()?.[0]?.toUpperCase() ?? "";
  return first + second || "?";
}

interface FarmerAvatarProps {
  firstName?: string | null;
  middleName?: string | null;
  uid: string;
  size?: "sm" | "md" | "lg";
}

const SIZES = {
  sm: { container: 32, font: 12, ring: 1 },
  md: { container: 48, font: 17, ring: 2 },
  lg: { container: 160, font: 52, ring: 3 },
} as const;

export default function FarmerAvatar({ firstName, middleName, uid, size = "md" }: FarmerAvatarProps) {
  const initials = getInitials(firstName, middleName);
  const color = PALETTE[hashCode(uid) % PALETTE.length];
  const s = SIZES[size];
  const half = s.container / 2;

  return (
    <svg
      width={s.container}
      height={s.container}
      viewBox={`0 0 ${s.container} ${s.container}`}
      role="img"
      aria-label={`Avatar for ${firstName ?? uid}`}
      className="shrink-0"
    >
      <circle
        cx={half}
        cy={half}
        r={half - s.ring}
        fill={color.bg}
        stroke={color.bg}
        strokeWidth={s.ring}
        strokeOpacity={0.3}
      />
      <text
        x="50%"
        y="50%"
        dy=".1em"
        textAnchor="middle"
        dominantBaseline="central"
        fill={color.text}
        fontSize={s.font}
        fontWeight={600}
        fontFamily="var(--font-sans), system-ui, sans-serif"
      >
        {initials}
      </text>
    </svg>
  );
}
