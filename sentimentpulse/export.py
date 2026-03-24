"""
SentimentPulse - Export utilities
Built by Himal Badu, AI Founder

Utilities for exporting analysis results in various formats.
"""

import csv
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from loguru import logger


class BaseExporter:
    """Base class for exporters."""
    
    def export(self, results: List[Dict[str, Any]], filepath: str):
        """Export results to file."""
        raise NotImplementedError


class JSONExporter(BaseExporter):
    """Export results as JSON."""
    
    def export(self, results: List[Dict[str, Any]], filepath: str):
        """Export results as JSON file."""
        data = {
            "exported_at": datetime.utcnow().isoformat(),
            "total_results": len(results),
            "results": results
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(results)} results to {filepath}")


class CSVExporter(BaseExporter):
    """Export results as CSV."""
    
    def export(self, results: List[Dict[str, Any]], filepath: str):
        """Export results as CSV file."""
        if not results:
            logger.warning("No results to export")
            return
        
        # Flatten results for CSV
        fieldnames = ["text", "sentiment", "score", "confidence", "model", "analyzed_at"]
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    "text": result.get("text", ""),
                    "sentiment": result.get("sentiment", ""),
                    "score": result.get("score", 0),
                    "confidence": result.get("confidence", 0),
                    "model": result.get("model", ""),
                    "analyzed_at": result.get("analyzed_at", "")
                }
                writer.writerow(row)
        
        logger.info(f"Exported {len(results)} results to {filepath}")


class XMLExporter(BaseExporter):
    """Export results as XML."""
    
    def export(self, results: List[Dict[str, Any]], filepath: str):
        """Export results as XML file."""
        root = ET.Element("sentiment_analysis")
        root.set("exported_at", datetime.utcnow().isoformat())
        root.set("total", str(len(results)))
        
        for i, result in enumerate(results, 1):
            item = ET.SubElement(root, "result")
            item.set("id", str(i))
            
            # Add text
            text_elem = ET.SubElement(item, "text")
            text_elem.text = result.get("text", "")
            
            # Add sentiment
            sentiment_elem = ET.SubElement(item, "sentiment")
            sentiment_elem.text = result.get("sentiment", "")
            
            # Add score
            score_elem = ET.SubElement(item, "score")
            score_elem.text = str(result.get("score", 0))
            
            # Add confidence
            conf_elem = ET.SubElement(item, "confidence")
            conf_elem.text = str(result.get("confidence", 0))
            
            # Add model
            model_elem = ET.SubElement(item, "model")
            model_elem.text = result.get("model", "")
            
            # Add timestamp
            time_elem = ET.SubElement(item, "analyzed_at")
            time_elem.text = result.get("analyzed_at", "")
        
        tree = ET.ElementTree(root)
        tree.write(filepath, encoding="utf-8", xml_declaration=True)
        
        logger.info(f"Exported {len(results)} results to {filepath}")


class TextExporter(BaseExporter):
    """Export results as plain text."""
    
    def export(self, results: List[Dict[str, Any]], filepath: str):
        """Export results as text file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("SentimentPulse Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Exported: {datetime.utcnow().isoformat()}\n")
            f.write(f"Total Results: {len(results)}\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"{i}. {result.get('text', '')}\n")
                f.write(f"   Sentiment: {result.get('sentiment', '')}\n")
                f.write(f"   Score: {result.get('score', 0):.4f}\n")
                f.write(f"   Confidence: {result.get('confidence', 0)*100:.1f}%\n\n")
        
        logger.info(f"Exported {len(results)} results to {filepath}")


class MarkdownExporter(BaseExporter):
    """Export results as Markdown."""
    
    def export(self, results: List[Dict[str, Any]], filepath: str):
        """Export results as Markdown file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# SentimentPulse Analysis Results\n\n")
            f.write(f"**Exported:** {datetime.utcnow().isoformat()}\n\n")
            f.write(f"**Total Results:** {len(results)}\n\n")
            
            # Summary
            sentiments = [r.get("sentiment") for r in results]
            positive = sentiments.count("positive")
            negative = sentiments.count("negative")
            neutral = sentiments.count("neutral")
            
            f.write("## Summary\n\n")
            f.write(f"- Positive: {positive} ({positive/len(results)*100:.1f}%)\n")
            f.write(f"- Negative: {negative} ({negative/len(results)*100:.1f}%)\n")
            f.write(f"- Neutral: {neutral} ({neutral/len(results)*100:.1f}%)\n\n")
            
            # Results table
            f.write("## Results\n\n")
            f.write("| # | Text | Sentiment | Score | Confidence |\n")
            f.write("|---|------|-----------|-------|------------|\n")
            
            for i, result in enumerate(results, 1):
                text = result.get("text", "")[:30]
                sentiment = result.get("sentiment", "")
                score = result.get("score", 0)
                confidence = result.get("confidence", 0) * 100
                
                f.write(f"| {i} | {text} | {sentiment} | {score:.3f} | {confidence:.1f}% |\n")
        
        logger.info(f"Exported {len(results)} results to {filepath}")


class ExportManager:
    """Manager for exporting results in various formats."""
    
    EXPORTERS = {
        "json": JSONExporter,
        "csv": CSVExporter,
        "xml": XMLExporter,
        "txt": TextExporter,
        "md": MarkdownExporter,
        "markdown": MarkdownExporter
    }
    
    @classmethod
    def export(
        cls,
        results: List[Dict[str, Any]],
        filepath: str,
        format: str = None
    ):
        """Export results to file.
        
        Args:
            results: List of analysis results
            filepath: Output file path
            format: Export format (json, csv, xml, txt, md)
        """
        if format is None:
            # Detect format from file extension
            format = Path(filepath).suffix.lstrip(".").lower()
        
        if format not in cls.EXPORTERS:
            raise ValueError(f"Unsupported format: {format}")
        
        exporter = cls.EXPORTERS[format]()
        exporter.export(results, filepath)
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of supported export formats."""
        return list(cls.EXPORTERS.keys())
