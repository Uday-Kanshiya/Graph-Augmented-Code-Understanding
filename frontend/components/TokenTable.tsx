import type { TokenMeasurement } from "@/lib/types";

export function TokenTable({ stages }: { stages: Record<string, TokenMeasurement> }) {
  const entries = Object.entries(stages);
  if (!entries.length) {
    return <p className="text-sm text-zinc-500">No token measurements yet.</p>;
  }

  return (
    <div className="overflow-hidden border border-line bg-white">
      <table className="w-full border-collapse text-left text-sm">
        <thead className="bg-panel text-xs uppercase tracking-wide text-zinc-500">
          <tr>
            <th className="px-3 py-2">Stage</th>
            <th className="px-3 py-2">Tokens</th>
            <th className="px-3 py-2">Count</th>
            <th className="px-3 py-2">Provider</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([key, value]) => (
            <tr key={key} className="border-t border-line">
              <td className="px-3 py-2 font-medium text-ink">{value.stage}</td>
              <td className="px-3 py-2 tabular-nums">{value.tokens.toLocaleString()}</td>
              <td className="px-3 py-2">{value.count_type}</td>
              <td className="px-3 py-2 text-zinc-600">{value.provider ?? value.model ?? "local"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

