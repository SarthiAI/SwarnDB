// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! A small, correct, dependency-free LRU cache used for the hot adjacency cache
//! in the typed graph store. O(1) get/put/invalidate via an intrusive doubly
//! linked list stored over a slab `Vec` (no `unsafe`, no external crate).

use std::collections::HashMap;
use std::hash::Hash;

struct Entry<K, V> {
    key: K,
    val: V,
    prev: Option<usize>,
    next: Option<usize>,
}

/// A bounded least-recently-used cache. Capacity is fixed at construction and
/// is at least 1. Eviction drops the least-recently-used entry on overflow.
pub struct LruCache<K, V> {
    cap: usize,
    map: HashMap<K, usize>,
    slots: Vec<Option<Entry<K, V>>>,
    free: Vec<usize>,
    head: Option<usize>, // most recently used
    tail: Option<usize>, // least recently used
    len: usize,
}

impl<K: Clone + Eq + Hash, V> LruCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            cap: capacity.max(1),
            map: HashMap::new(),
            slots: Vec::new(),
            free: Vec::new(),
            head: None,
            tail: None,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.slots.clear();
        self.free.clear();
        self.head = None;
        self.tail = None;
        self.len = 0;
    }

    /// Unlink slot `idx` from the recency list (does not free it).
    fn detach(&mut self, idx: usize) {
        let (prev, next) = {
            let e = self.slots[idx].as_ref().expect("detach live slot");
            (e.prev, e.next)
        };
        match prev {
            Some(p) => self.slots[p].as_mut().unwrap().next = next,
            None => self.head = next,
        }
        match next {
            Some(n) => self.slots[n].as_mut().unwrap().prev = prev,
            None => self.tail = prev,
        }
        let e = self.slots[idx].as_mut().unwrap();
        e.prev = None;
        e.next = None;
    }

    /// Link slot `idx` at the front (most recently used).
    fn push_front(&mut self, idx: usize) {
        let old_head = self.head;
        {
            let e = self.slots[idx].as_mut().unwrap();
            e.prev = None;
            e.next = old_head;
        }
        if let Some(h) = old_head {
            self.slots[h].as_mut().unwrap().prev = Some(idx);
        }
        self.head = Some(idx);
        if self.tail.is_none() {
            self.tail = Some(idx);
        }
    }

    /// Fetch a value, marking it most-recently-used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(&idx) = self.map.get(key) {
            self.detach(idx);
            self.push_front(idx);
            return self.slots[idx].as_ref().map(|e| &e.val);
        }
        None
    }

    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Insert or update a value, marking it most-recently-used. Evicts the LRU
    /// entry if inserting a new key would exceed capacity.
    pub fn put(&mut self, key: K, val: V) {
        if let Some(&idx) = self.map.get(&key) {
            self.slots[idx].as_mut().unwrap().val = val;
            self.detach(idx);
            self.push_front(idx);
            return;
        }

        if self.len >= self.cap {
            if let Some(t) = self.tail {
                self.detach(t);
                if let Some(old) = self.slots[t].take() {
                    self.map.remove(&old.key);
                }
                self.free.push(t);
                self.len -= 1;
            }
        }

        let idx = if let Some(f) = self.free.pop() {
            self.slots[f] = Some(Entry {
                key: key.clone(),
                val,
                prev: None,
                next: None,
            });
            f
        } else {
            self.slots.push(Some(Entry {
                key: key.clone(),
                val,
                prev: None,
                next: None,
            }));
            self.slots.len() - 1
        };
        self.map.insert(key, idx);
        self.push_front(idx);
        self.len += 1;
    }

    /// Drop a key if present.
    pub fn invalidate(&mut self, key: &K) {
        if let Some(idx) = self.map.remove(key) {
            self.detach(idx);
            self.slots[idx] = None;
            self.free.push(idx);
            self.len -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_least_recently_used() {
        let mut c: LruCache<u64, u64> = LruCache::new(2);
        c.put(1, 10);
        c.put(2, 20);
        assert_eq!(c.get(&1).copied(), Some(10)); // 1 now MRU, 2 is LRU
        c.put(3, 30); // evicts 2
        assert_eq!(c.get(&2), None);
        assert_eq!(c.get(&1).copied(), Some(10));
        assert_eq!(c.get(&3).copied(), Some(30));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn update_existing_does_not_grow() {
        let mut c: LruCache<u64, u64> = LruCache::new(2);
        c.put(1, 10);
        c.put(1, 11);
        assert_eq!(c.len(), 1);
        assert_eq!(c.get(&1).copied(), Some(11));
    }

    #[test]
    fn invalidate_removes() {
        let mut c: LruCache<u64, u64> = LruCache::new(4);
        c.put(1, 10);
        c.put(2, 20);
        c.invalidate(&1);
        assert_eq!(c.get(&1), None);
        assert_eq!(c.len(), 1);
        // slot is reused on next insert
        c.put(3, 30);
        assert_eq!(c.get(&3).copied(), Some(30));
    }
}
