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
  predictionModalOpen: boolean;
  setActiveJobId: (id: string | null) => void;
  clearActiveJobId: () => void;
  openPredictionModal: () => void;
  closePredictionModal: () => void;
}

export const useJobStore = create<JobState>()((set) => ({
  activeJobId: null,
  predictionModalOpen: false,
  setActiveJobId: (id) => set({ activeJobId: id }),
  clearActiveJobId: () => set({ activeJobId: null }),
  openPredictionModal: () => set({ predictionModalOpen: true }),
  closePredictionModal: () => set({ predictionModalOpen: false }),
}));
