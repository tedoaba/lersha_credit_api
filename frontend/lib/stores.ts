/**
 * Zustand stores for client-global session state.
 *
 * useJobStore — session-only (no persistence).
 *   Holds the currently active polling job ID after a prediction is submitted.
 *
 * NOTE: The API key is managed server-side via the API_KEY environment variable.
 * There is no API key store in the browser.
 */

import { create } from "zustand";

interface JobState {
  activeJobId: string | null;
  setActiveJobId: (id: string | null) => void;
  clearActiveJobId: () => void;
}

export const useJobStore = create<JobState>()((set) => ({
  activeJobId: null,
  setActiveJobId: (id) => set({ activeJobId: id }),
  clearActiveJobId: () => set({ activeJobId: null }),
}));
