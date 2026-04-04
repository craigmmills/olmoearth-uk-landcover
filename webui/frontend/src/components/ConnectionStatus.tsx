interface ConnectionStatusProps {
  healthy: boolean | null;
}

export default function ConnectionStatus({ healthy }: ConnectionStatusProps) {
  const color =
    healthy === null ? 'bg-yellow-400' : healthy ? 'bg-green-500' : 'bg-red-500';
  const label =
    healthy === null ? 'Checking...' : healthy ? 'Backend connected' : 'Backend offline';

  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <span className={`inline-block w-2 h-2 rounded-full ${color}`} />
      {label}
    </div>
  );
}
