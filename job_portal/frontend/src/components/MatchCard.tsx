import type { JobMatch, SkillStatus } from '../types';

const CHIP_CLASS: Record<SkillStatus, string> = {
  Have: 'chip have',
  Transferable: 'chip transferable',
  Missing: 'chip missing',
};

/**
 * One scored match.
 *
 * Every skill is shown with its verdict, not just the gaps. A candidate looking
 * at a 65% wants to know what the other 35% was, and a card that lists only what
 * is missing reads as a rejection rather than as advice.
 */
export function MatchCard({ match }: { match: JobMatch }) {
  const { job } = match;

  return (
    <article className="card job-card">
      <div className="fit">
        {match.fitScore.toFixed(1)}%
        <span className="band">{match.fitBand}</span>
      </div>
      <div className="bar"><span style={{ width: `${Math.min(100, match.fitScore)}%` }} /></div>

      <h3>{job.title}</h3>
      <div className="meta">
        {job.company && <span>{job.company}</span>}
        {job.location && <span>· {job.location}</span>}
        <span>· {job.workMode}</span>
        {job.seniorityLevel !== 'Unspecified' && <span>· {job.seniorityLevel}</span>}
      </div>

      <div className="breakdown">
        <div><b>{match.semanticScore.toFixed(0)}%</b>relevance</div>
        <div><b>{match.skillScore.toFixed(0)}%</b>skills</div>
        <div><b>{match.titleScore.toFixed(0)}%</b>title</div>
        <div><b>{match.experienceScore.toFixed(0)}%</b>experience</div>
      </div>

      {match.skills.length > 0 && (
        <div className="chips">
          {match.skills.map(skill => (
            <span
              key={skill.skill}
              className={CHIP_CLASS[skill.status] ?? 'chip'}
              /* The evidence belongs on the chip: "Transferable" on its own
                 invites "transferable from what?", and the answer is already known. */
              title={
                skill.status === 'Missing'
                  ? 'Not evidenced on your resume'
                  : `Matched by your ${skill.evidenceSkill} (${(skill.similarity * 100).toFixed(0)}%)`
              }
            >
              {skill.skill}
            </span>
          ))}
        </div>
      )}

      <p className="note">{match.recruiterNote}</p>

      {job.applyUrl && (
        <a href={job.applyUrl} target="_blank" rel="noreferrer">Apply →</a>
      )}
    </article>
  );
}
