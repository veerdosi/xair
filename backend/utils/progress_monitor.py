"""
Progress monitoring utility for tracking long-running processes.
Provides estimated time remaining for multi-stage operations.
"""

import time
import logging
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum, auto

logger = logging.getLogger(__name__)

class Stage(Enum):
    """Enum representing different processing stages."""
    INIT = auto()
    CGRT_GENERATION = auto()
    CGRT_DIVERGENCE = auto()
    CGRT_TREE_BUILDING = auto()
    CGRT_ATTENTION = auto()
    COUNTERFACTUAL_GENERATION = auto()
    COUNTERFACTUAL_EVALUATION = auto()
    KG_ENTITY_MAPPING = auto()
    KG_VALIDATION = auto()
    VISUALIZATION = auto()
    COMPLETE = auto()

class ProgressMonitor:
    """
    Progress monitoring utility for tracking multi-stage processes.
    Estimates time remaining based on previous runs and current progress.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the progress monitor.

        Args:
            verbose: Whether to print progress updates
        """
        self.verbose = verbose
        self.current_stage = Stage.INIT
        self.start_time = time.time()
        self.stage_start_times: Dict[Stage, float] = {}
        self.stage_durations: Dict[Stage, float] = {}
        self.estimated_durations: Dict[Stage, float] = {
            # Initial estimates based on typical runs (in seconds)
            Stage.INIT: 1.0,
            Stage.CGRT_GENERATION: 15.0,
            Stage.CGRT_DIVERGENCE: 5.0,
            Stage.CGRT_TREE_BUILDING: 3.0,
            Stage.CGRT_ATTENTION: 10.0,
            Stage.COUNTERFACTUAL_GENERATION: 20.0,
            Stage.COUNTERFACTUAL_EVALUATION: 5.0,
            Stage.KG_ENTITY_MAPPING: 10.0,
            Stage.KG_VALIDATION: 5.0,
            Stage.VISUALIZATION: 3.0
        }
        self.completed_stages: List[Stage] = []

        # Register start of initialization
        self.start_stage(Stage.INIT)

    def start_stage(self, stage: Stage):
        """
        Start a new processing stage.

        Args:
            stage: The stage being started
        """
        self.current_stage = stage
        self.stage_start_times[stage] = time.time()

        if self.verbose:
            # Calculate estimated time for this stage
            est_duration = self.estimated_durations.get(stage, 0)

            # Calculate estimated total time remaining
            remaining = self.estimate_remaining_time()

            logger.info(f"Starting {stage.name} (est. {est_duration:.1f}s, total remaining: {remaining:.1f}s)")

    def complete_stage(self, stage: Stage):
        """
        Mark a stage as completed and update timing estimates.

        Args:
            stage: The stage being completed
        """
        if stage in self.stage_start_times:
            end_time = time.time()
            duration = end_time - self.stage_start_times[stage]
            self.stage_durations[stage] = duration
            self.completed_stages.append(stage)

            # Update the estimated duration based on actual performance
            # Use a weighted average to adapt but maintain stability
            if stage in self.estimated_durations:
                # 70% old estimate, 30% new actual - adjust as needed
                self.estimated_durations[stage] = (
                    0.7 * self.estimated_durations[stage] +
                    0.3 * duration
                )
            else:
                self.estimated_durations[stage] = duration

            if self.verbose:
                logger.info(f"Completed {stage.name} in {duration:.2f}s")

        # Move to next stage if this was the current one
        if stage == self.current_stage:
            next_stages = {
                Stage.INIT: Stage.CGRT_GENERATION,
                Stage.CGRT_GENERATION: Stage.CGRT_DIVERGENCE,
                Stage.CGRT_DIVERGENCE: Stage.CGRT_TREE_BUILDING,
                Stage.CGRT_TREE_BUILDING: Stage.CGRT_ATTENTION,
                Stage.CGRT_ATTENTION: Stage.COUNTERFACTUAL_GENERATION,
                Stage.COUNTERFACTUAL_GENERATION: Stage.COUNTERFACTUAL_EVALUATION,
                Stage.COUNTERFACTUAL_EVALUATION: Stage.KG_ENTITY_MAPPING,
                Stage.KG_ENTITY_MAPPING: Stage.KG_VALIDATION,
                Stage.KG_VALIDATION: Stage.VISUALIZATION,
                Stage.VISUALIZATION: Stage.COMPLETE
            }

            if stage in next_stages:
                self.current_stage = next_stages[stage]

    def update_progress(self, stage: Stage, progress: float):
        """
        Update progress within a stage.

        Args:
            stage: The current stage
            progress: Progress value between 0 and 1
        """
        if self.verbose and stage == self.current_stage:
            est_duration = self.estimated_durations.get(stage, 0)
            elapsed = time.time() - self.stage_start_times.get(stage, time.time())

            # Estimate time remaining for current stage
            if progress > 0:
                stage_remaining = (est_duration * (1 - progress)) if progress < 1 else 0
            else:
                stage_remaining = est_duration

            # Calculate total estimated time remaining
            total_remaining = self.estimate_remaining_time()

            logger.info(f"{stage.name}: {progress*100:.1f}% complete (est. {stage_remaining:.1f}s stage, {total_remaining:.1f}s total remaining)")

    def estimate_remaining_time(self) -> float:
        """
        Estimate the total time remaining.

        Returns:
            Estimated time remaining in seconds
        """
        remaining_time = 0.0

        # Add estimated time for current stage
        if self.current_stage != Stage.COMPLETE:
            if self.current_stage in self.stage_start_times:
                elapsed = time.time() - self.stage_start_times[self.current_stage]
                est_duration = self.estimated_durations.get(self.current_stage, 0)
                remaining_time += max(0, est_duration - elapsed)
            else:
                remaining_time += self.estimated_durations.get(self.current_stage, 0)

        # Add estimated time for future stages
        all_stages = list(Stage)
        current_index = all_stages.index(self.current_stage)

        # Add times for all stages after the current one
        for stage in all_stages[current_index+1:]:
            if stage != Stage.COMPLETE and stage not in self.completed_stages:
                remaining_time += self.estimated_durations.get(stage, 0)

        return remaining_time

    def get_elapsed_time(self) -> float:
        """
        Get the total elapsed time.

        Returns:
            Total elapsed time in seconds
        """
        return time.time() - self.start_time

    def get_stage_summary(self) -> Dict[str, float]:
        """
        Get a summary of time spent in each stage.

        Returns:
            Dictionary mapping stage names to durations
        """
        return {stage.name: duration for stage, duration in self.stage_durations.items()}

    def get_progress_report(self) -> Dict[str, Any]:
        """
        Get a detailed progress report.

        Returns:
            Dictionary with progress information
        """
        return {
            "elapsed_time": self.get_elapsed_time(),
            "current_stage": self.current_stage.name,
            "completed_stages": [stage.name for stage in self.completed_stages],
            "stage_durations": self.get_stage_summary(),
            "estimated_remaining": self.estimate_remaining_time()
        }

    def complete(self):
        """Mark the entire process as complete."""
        self.complete_stage(self.current_stage)
        self.current_stage = Stage.COMPLETE

        if self.verbose:
            total_time = self.get_elapsed_time()
            logger.info(f"Process completed in {total_time:.2f}s")

            # Show time spent in each stage
            stage_summary = self.get_stage_summary()
            for stage_name, duration in stage_summary.items():
                percentage = (duration / total_time) * 100
                logger.info(f"  {stage_name}: {duration:.2f}s ({percentage:.1f}%)")
