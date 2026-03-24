"""
SentimentPulse - Pipeline for batch processing
Built by Himal Badu, AI Founder

Pipeline for processing large batches of texts efficiently.
"""

import time
from typing import List, Dict, Any, Callable, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class ProcessingMode(Enum):
    """Processing mode for pipelines."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    STREAMING = "streaming"


@dataclass
class PipelineConfig:
    """Configuration for processing pipeline."""
    mode: ProcessingMode = ProcessingMode.PARALLEL
    batch_size: int = 32
    max_workers: int = 4
    progress_callback: Optional[Callable[[int, int], None]] = None


class ProcessingPipeline:
    """Pipeline for processing texts through sentiment analysis."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._results: List[Dict[str, Any]] = []
        self._errors: List[Dict[str, Any]] = []
    
    def process(
        self,
        texts: List[str],
        analyzer_func: Callable[[str], Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process texts through the pipeline.
        
        Args:
            texts: List of texts to process
            analyzer_func: Function to analyze each text
            
        Returns:
            List of analysis results
        """
        self._results.clear()
        self._errors.clear()
        
        total = len(texts)
        logger.info(f"Starting pipeline processing: {total} texts")
        
        if self.config.mode == ProcessingMode.SEQUENTIAL:
            self._process_sequential(texts, analyzer_func)
        elif self.config.mode == ProcessingMode.PARALLEL:
            self._process_parallel(texts, analyzer_func)
        elif self.config.mode == ProcessingMode.STREAMING:
            return self._process_streaming(texts, analyzer_func)
        
        return self._results
    
    def _process_sequential(
        self,
        texts: List[str],
        analyzer_func: Callable[[str], Dict[str, Any]]
    ):
        """Process texts sequentially."""
        total = len(texts)
        
        for i, text in enumerate(texts):
            try:
                result = analyzer_func(text)
                result["text"] = text
                self._results.append(result)
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                self._errors.append({
                    "index": i,
                    "text": text,
                    "error": str(e)
                })
            
            if self.config.progress_callback:
                self.config.progress_callback(i + 1, total)
    
    def _process_parallel(
        self,
        texts: List[str],
        analyzer_func: Callable[[str], Dict[str, Any]]
    ):
        """Process texts in parallel."""
        total = len(texts)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_index = {
                executor.submit(analyzer_func, text): i
                for i, text in enumerate(texts)
            }
            
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    result["text"] = texts[index]
                    self._results.append(result)
                except Exception as e:
                    logger.error(f"Error processing text {index}: {e}")
                    self._errors.append({
                        "index": index,
                        "text": texts[index],
                        "error": str(e)
                    })
                
                completed += 1
                if self.config.progress_callback:
                    self.config.progress_callback(completed, total)
    
    def _process_streaming(
        self,
        texts: List[str],
        analyzer_func: Callable[[str], Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Process texts in streaming mode."""
        total = len(texts)
        
        for i, text in enumerate(texts):
            try:
                result = analyzer_func(text)
                result["text"] = text
                yield result
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                self._errors.append({
                    "index": i,
                    "text": text,
                    "error": str(e)
                })
            
            if self.config.progress_callback:
                self.config.progress_callback(i + 1, total)
    
    def process_batches(
        self,
        texts: List[str],
        analyzer_func: Callable[[List[str]], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Process texts in batches.
        
        Args:
            texts: List of texts to process
            analyzer_func: Function to analyze batch of texts
            
        Returns:
            List of analysis results
        """
        self._results.clear()
        self._errors.clear()
        
        total = len(texts)
        batch_size = self.config.batch_size
        
        logger.info(f"Starting batch processing: {total} texts, batch size: {batch_size}")
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                results = analyzer_func(batch)
                
                for text, result in zip(batch, results):
                    result["text"] = text
                    self._results.append(result)
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {e}")
                for j, text in enumerate(batch):
                    self._errors.append({
                        "index": i + j,
                        "text": text,
                        "error": str(e)
                    })
            
            if self.config.progress_callback:
                self.config.progress_callback(min(i + batch_size, total), total)
        
        return self._results
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get processing results."""
        return self._results.copy()
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """Get processing errors."""
        return self._errors.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_processed": len(self._results),
            "total_errors": len(self._errors),
            "success_rate": (
                len(self._results) / (len(self._results) + len(self._errors)) * 100
                if (len(self._results) + len(self._errors)) > 0 else 0
            )
        }


class StreamingPipeline(ProcessingPipeline):
    """Streaming pipeline for real-time processing."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(config)
        self.config.mode = ProcessingMode.STREAMING
    
    def process_stream(
        self,
        text_iterator: Iterator[str],
        analyzer_func: Callable[[str], Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Process text stream.
        
        Args:
            text_iterator: Iterator of texts
            analyzer_func: Function to analyze each text
            
        Yields:
            Analysis results
        """
        self._errors.clear()
        
        for text in text_iterator:
            try:
                result = analyzer_func(text)
                result["text"] = text
                yield result
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                self._errors.append({
                    "text": text,
                    "error": str(e)
                })
