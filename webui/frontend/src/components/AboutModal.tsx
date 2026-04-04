import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog';

interface AboutModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export default function AboutModal({ open, onOpenChange }: AboutModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogTitle>About UK Landcover</DialogTitle>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            Self-correcting landcover classification pipeline for the UK using satellite
            imagery and iterative machine learning refinement.
          </p>
          <h3 className="font-semibold text-foreground">Methodology</h3>
          <ul className="list-disc pl-4 space-y-1">
            <li>Sentinel-2 satellite imagery at 10m resolution</li>
            <li>Random Forest classification with 6 landcover classes</li>
            <li>Iterative refinement guided by Gemini VLM evaluation</li>
            <li>Validated against ESA WorldCover 2021</li>
          </ul>
          <h3 className="font-semibold text-foreground">Classes</h3>
          <p>Built-up, Cropland, Grassland, Tree cover, Water, Other</p>
          <h3 className="font-semibold text-foreground">Keyboard Shortcuts</h3>
          <ul className="space-y-1 font-mono text-xs">
            <li><kbd className="rounded bg-muted px-1.5 py-0.5">L</kbd> Toggle side panel</li>
            <li><kbd className="rounded bg-muted px-1.5 py-0.5">D</kbd> Toggle dashboard</li>
            <li><kbd className="rounded bg-muted px-1.5 py-0.5">C</kbd> Toggle comparison</li>
          </ul>
        </div>
      </DialogContent>
    </Dialog>
  );
}
