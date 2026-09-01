/**
 * The colour a fit percentage is drawn in.
 *
 * Split out of `MatchVisuals` so that file exports only components — Fast Refresh
 * cannot handle a module that mixes the two, the same reason `portal-context`
 * lives apart from its provider.
 *
 * One set of bands for the whole portal, so a colour means the same thing on
 * every surface. The two scorers must agree here in particular: a 62 has to read
 * identically whether it was computed or reasoned, or the mode switch would look
 * like it changed the quality of the role rather than the way it was judged.
 */
export const scoreTone = (score: number) =>
  score >= 80
    ? {
        text: 'text-emerald-400',
        ring: 'stroke-emerald-400',
        bg: 'bg-emerald-500',
        soft: 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300',
      }
    : score >= 60
      ? {
          text: 'text-sky-400',
          ring: 'stroke-sky-400',
          bg: 'bg-sky-500',
          soft: 'border-sky-500/20 bg-sky-500/10 text-sky-300',
        }
      : score >= 35
        ? {
            text: 'text-amber-400',
            ring: 'stroke-amber-400',
            bg: 'bg-amber-500',
            soft: 'border-amber-500/20 bg-amber-500/10 text-amber-300',
          }
        : {
            text: 'text-rose-400',
            ring: 'stroke-rose-400',
            bg: 'bg-rose-500',
            soft: 'border-rose-500/20 bg-rose-500/10 text-rose-300',
          };
