"""Stream plugins. Each module defines one Stream subclass and registers it in STREAMS."""
from __future__ import annotations


def registry() -> dict:
    """name -> Stream class. Imported lazily so a broken plugin can't break the others."""
    from . import briefing, grants, journal_club, literature, people, podcast, reporter, trials, tuning
    out = {"literature": literature.LiteratureStream, "grants": grants.GrantsStream,
           "reporter": reporter.ReporterStream, "trials": trials.TrialsStream,
           "journal_club": journal_club.JournalClubStream,
           "people": people.PeopleStream, "briefing": briefing.BriefingStream,
           "tuning": tuning.TuningStream, "podcast": podcast.PodcastStream}
    return out
