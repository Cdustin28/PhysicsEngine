//! Snapshot-based undo/redo.
//!
//! Rather than recording every kind of edit as a command, the editor keeps
//! copies of the whole scene. After each frame in which the user isn't in
//! the middle of something (no mouse button held, simulation paused), it
//! compares the scene with the last recorded one; if they differ, that's one
//! undoable edit. A whole drag is therefore one undo step, and any new kind of
//! edit is undoable automatically.
//!
//! Scenes are small (fluid scratch space isn't copied; see
//! `engine::Fluid`), so keeping a hundred copies is cheap.

/// Undo/redo stacks around a "committed" state: the last state recorded.
#[derive(Debug, Clone)]
pub struct History<T> {
    undo: Vec<T>,
    redo: Vec<T>,
    committed: Option<T>,
    limit: usize,
}

impl<T: Clone + PartialEq> History<T> {
    /// A history that remembers at most `limit` undo steps.
    pub fn new(limit: usize) -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            committed: None,
            limit,
        }
    }

    /// Make `state` the baseline without recording an edit, e.g. after the
    /// simulation ran (its progress isn't an edit to undo).
    pub fn rebase(&mut self, state: &T) {
        self.committed = Some(state.clone());
    }

    /// If `state` differs from the last recorded state, record the change as
    /// one undoable edit. Returns whether anything was recorded.
    pub fn record(&mut self, state: &T) -> bool {
        match &self.committed {
            None => {
                self.committed = Some(state.clone());
                false
            }
            Some(previous) if previous == state => false,
            Some(_) => {
                let previous = self
                    .committed
                    .replace(state.clone())
                    .expect("checked above");
                self.undo.push(previous);
                if self.undo.len() > self.limit {
                    self.undo.remove(0);
                }
                self.redo.clear();
                true
            }
        }
    }

    /// Step back one edit, returning the state to restore.
    pub fn undo(&mut self) -> Option<T> {
        let previous = self.undo.pop()?;
        if let Some(current) = self.committed.replace(previous.clone()) {
            self.redo.push(current);
        }
        Some(previous)
    }

    /// Step forward one undone edit, returning the state to restore.
    pub fn redo(&mut self) -> Option<T> {
        let next = self.redo.pop()?;
        if let Some(current) = self.committed.replace(next.clone()) {
            self.undo.push(current);
        }
        Some(next)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_only_changes() {
        let mut h = History::new(10);
        assert!(!h.record(&1)); // first call just sets the baseline
        assert!(!h.record(&1));
        assert!(h.record(&2));
        assert!(h.can_undo());
    }

    #[test]
    fn undo_and_redo_walk_the_history() {
        let mut h = History::new(10);
        for s in [1, 2, 3] {
            h.record(&s);
        }
        assert_eq!(h.undo(), Some(2));
        assert_eq!(h.undo(), Some(1));
        assert_eq!(h.undo(), None);
        assert_eq!(h.redo(), Some(2));
        assert_eq!(h.redo(), Some(3));
        assert_eq!(h.redo(), None);
    }

    #[test]
    fn a_new_edit_clears_redo() {
        let mut h = History::new(10);
        for s in [1, 2, 3] {
            h.record(&s);
        }
        h.undo(); // back to 2
        assert!(h.record(&5));
        assert!(!h.can_redo());
        assert_eq!(h.undo(), Some(2));
    }

    #[test]
    fn rebase_is_not_an_edit() {
        let mut h = History::new(10);
        h.record(&1);
        h.record(&2);
        h.rebase(&7); // e.g. the simulation ran
        assert!(!h.record(&7));
        assert_eq!(h.undo(), Some(1));
        assert_eq!(h.redo(), Some(7));
    }

    #[test]
    fn limit_drops_the_oldest_steps() {
        let mut h = History::new(2);
        for s in 0..5 {
            h.record(&s);
        }
        assert_eq!(h.undo(), Some(3));
        assert_eq!(h.undo(), Some(2));
        assert_eq!(h.undo(), None);
    }
}
