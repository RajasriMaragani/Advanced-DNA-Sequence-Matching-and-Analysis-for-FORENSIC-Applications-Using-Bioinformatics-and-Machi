import pandas as pd
import numpy as np
from Bio import SeqIO
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import BertModel, BertConfig, BertTokenizer, AutoModel, AutoTokenizer
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GenomicDataHandler:
    """Handler for genomic data retrieval and processing."""
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize the GenomicDataHandler.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.species_datasets = {
            'human': {
                'expression': 'https://www.encodeproject.org/files/ENCFF835NTC/@@download/ENCFF835NTC.tsv',
                'regulatory': 'https://www.encodeproject.org/files/ENCFF678DIT/@@download/ENCFF678DIT.bed'
            },
            'arabidopsis': {
                'expression': 'https://www.arabidopsis.org/download_files/Public_Data_Releases/TAIR_Data_20201231/ExpressionAtlas/AtlasExpressionMatrix.txt',
                'regulatory': 'https://www.arabidopsis.org/download_files/Public_Data_Releases/TAIR_Data_20201231/Epigenome/epigenome_rrbs_sites.txt'
            },
            'mouse': {
                'expression': 'https://www.encodeproject.org/files/ENCFF726LXD/@@download/ENCFF726LXD.tsv',
                'regulatory': 'https://www.encodeproject.org/files/ENCFF212QQP/@@download/ENCFF212QQP.bed'
            }
        }
    
    def fetch_genome_sequence(self, species: str, chromosome: str, start: int, end: int) -> str:
        """
        Fetch genomic sequence from UCSC or Ensembl.
        
        Args:
            species: Species name (human, arabidopsis, etc.)
            chromosome: Chromosome name/number
            start: Start position
            end: End position
            
        Returns:
            Genomic sequence as string
            
        Raises:
            ValueError: If the sequence cannot be retrieved
        """
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"{species}_{chromosome}_{start}_{end}.fasta")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return f.read()
        
        # Fetch from API if not in cache
        try:
            if species == 'human':
                url = f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={chromosome};start={start};end={end}"
            elif species == 'mouse':
                url = f"https://api.genome.ucsc.edu/getData/sequence?genome=mm10;chrom={chromosome};start={start};end={end}"
            elif species == 'arabidopsis':
                url = f"http://plants.ensembl.org/Arabidopsis_thaliana/Export/Output/Sequence?db=core;flank3_display=0;flank5_display=0;output=fasta;r={chromosome}:{start}-{end};strand=feature;genomic=unmasked;_format=Text"
            else:
                raise ValueError(f"Species {species} not supported")
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Cache the response
                with open(cache_file, 'w') as f:
                    f.write(response.text)
                return response.text
            else:
                raise ValueError(f"Failed to fetch sequence. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching sequence: {e}")
            raise ValueError(f"Error fetching sequence: {e}")
    
    def load_expression_data(self, species: str) -> pd.DataFrame:
        """
        Load gene expression data for a species.
        
        Args:
            species: Species name
            
        Returns:
            DataFrame containing expression data
            
        Raises:
            ValueError: If species is not supported
        """
        if species not in self.species_datasets:
            raise ValueError(f"Species {species} not supported")
            
        cache_file = os.path.join(self.cache_dir, f"{species}_expression.csv")
        
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)
        
        url = self.species_datasets[species]['expression']
        try:
            if species == 'human' or species == 'mouse':
                df = pd.read_csv(url, sep='\t')
            elif species == 'arabidopsis':
                df = pd.read_csv(url, sep='\t', header=0, index_col=0)
            
            # Cache the data
            df.to_csv(cache_file, index=False)
            return df
        except Exception as e:
            logger.error(f"Error loading expression data: {e}")
            raise ValueError(f"Error loading expression data: {e}")
    
    def load_regulatory_data(self, species: str) -> pd.DataFrame:
        """
        Load regulatory elements data for a species.
        
        Args:
            species: Species name
            
        Returns:
            DataFrame containing regulatory data
            
        Raises:
            ValueError: If species is not supported
        """
        if species not in self.species_datasets:
            raise ValueError(f"Species {species} not supported")
            
        cache_file = os.path.join(self.cache_dir, f"{species}_regulatory.csv")
        
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)
        
        url = self.species_datasets[species]['regulatory']
        try:
            df = pd.read_csv(url, sep='\t', header=None)
            
            # Add column names for BED files
            if species == 'human' or species == 'mouse':
                df.columns = ['chrom', 'start', 'end', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb']
            
            # Cache the data
            df.to_csv(cache_file, index=False)
            return df
        except Exception as e:
            logger.error(f"Error loading regulatory data: {e}")
            raise ValueError(f"Error loading regulatory data: {e}")
    
    def process_sequence_data(self, fasta_file: str) -> List[str]:
        """
        Process FASTA file into sequences.
        
        Args:
            fasta_file: Path to FASTA file or file-like object
            
        Returns:
            List of sequences
        """
        sequences = []
        try:
            if isinstance(fasta_file, str) and os.path.isfile(fasta_file):
                with open(fasta_file) as handle:
                    for record in SeqIO.parse(handle, "fasta"):
                        sequences.append(str(record.seq))
            else:
                # Assume it's a file-like object
                for record in SeqIO.parse(fasta_file, "fasta"):
                    sequences.append(str(record.seq))
            return sequences
        except Exception as e:
            logger.error(f"Error processing sequence data: {e}")
            raise ValueError(f"Error processing sequence data: {e}")
    
    def kmer_tokenize(self, sequence: str, k: int = 6) -> List[str]:
        """
        Tokenize DNA sequence into k-mers.
        
        Args:
            sequence: DNA sequence
            k: k-mer size
            
        Returns:
            List of k-mers
        """
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


class NucleotideTransformer(nn.Module):
    """Nucleotide Transformer (NT) model for genomic sequence analysis."""
    
    def __init__(self, model_name: str = "armheb/DNA_bert_6", num_labels: int = 2):
        """
        Initialize the Nucleotide Transformer model.
        
        Args:
            model_name: Pre-trained model name
            num_labels: Number of output classes
        """
        super(NucleotideTransformer, self).__init__()
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            
        Returns:
            Logits for each class
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def get_attention_weights(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Get attention weights for visualization.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            
        Returns:
            Attention weights as numpy array
        """
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask, output_attentions=True)
            # outputs.attentions is a tuple of tensors, one for each layer
            # Each tensor has shape [batch_size, num_heads, seq_len, seq_len]
            attention = outputs.attentions[-1]  # Get the last layer's attention
            # Average across heads
            attention = attention.mean(dim=1)  # Shape: [batch_size, seq_len, seq_len]
            return attention.cpu().numpy()


class AgroNT(NucleotideTransformer):
    """Agro Nucleotide Transformer for plant genomes."""
    
    def __init__(self, model_name: str = "zhihan1996/DNA_bert_6", num_labels: int = 2):
        """
        Initialize the AgroNT model.
        
        Args:
            model_name: Pre-trained model name
            num_labels: Number of output classes
        """
        super().__init__(model_name, num_labels)
        # Additional plant-specific layers
        self.plant_specific = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Update classifier to use new dimension
        self.classifier = nn.Linear(128, num_labels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            
        Returns:
            Logits for each class
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        plant_features = self.plant_specific(pooled_output)
        logits = self.classifier(plant_features)
        return logits


class SegmentNT(nn.Module):
    """SegmentNT model with U-Net like architecture for genomic segmentation."""
    
    def __init__(self, backbone_model: str = "armheb/DNA_bert_6", num_classes: int = 5):
        """
        Initialize the SegmentNT model.
        
        Args:
            backbone_model: Pre-trained model name
            num_classes: Number of segment classes
        """
        super(SegmentNT, self).__init__()
        self.backbone = AutoModel.from_pretrained(backbone_model)
        hidden_size = self.backbone.config.hidden_size
        
        # U-Net like architecture
        self.down1 = nn.Sequential(
            nn.Conv1d(hidden_size, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.up2 = nn.Sequential(
            nn.Conv1d(64 + 256, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.final = nn.Conv1d(64, num_classes, kernel_size=1)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_model)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            
        Returns:
            Logits for each position and class
        """
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        x = sequence_output.permute(0, 2, 1)  # (batch_size, hidden_size, seq_len)
        
        # Down sampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        
        # Up sampling with skip connections
        u1 = self.up1(d2)
        
        # Ensure dimensions match for concatenation
        diff = d1.size(2) - u1.size(2)
        if diff > 0:
            u1 = torch.nn.functional.pad(u1, (0, diff))
        elif diff < 0:
            d1 = torch.nn.functional.pad(d1, (0, -diff))
            
        u2 = self.up2(torch.cat([u1, d1], dim=1))
        
        # Final layer
        logits = self.final(u2)
        return logits.permute(0, 2, 1)  # (batch_size, seq_len, num_classes)


class GenomicDataset(Dataset):
    """Dataset for genomic sequences and labels."""
    
    def __init__(self, sequences: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            sequences: List of DNA sequences
            labels: List of labels
            tokenizer: Tokenizer for the sequences
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class SegmentationDataset(Dataset):
    """Dataset for genomic segmentation tasks."""
    
    def __init__(self, sequences: List[str], segment_labels: List[List[int]], tokenizer, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            sequences: List of DNA sequences
            segment_labels: List of segment labels for each position
            tokenizer: Tokenizer for the sequences
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.segment_labels = segment_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input_ids, attention_mask, and segment_labels
        """
        sequence = self.sequences[idx]
        segment_label = self.segment_labels[idx]
        
        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Pad segment labels to match tokenized sequence length
        padded_labels = torch.zeros(self.max_length, dtype=torch.long)
        seq_len = min(len(segment_label), self.max_length - 2)  # Account for [CLS] and [SEP]
        padded_labels[1:seq_len+1] = torch.tensor(segment_label[:seq_len], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'segment_labels': padded_labels
        }


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                epochs: int = 10, learning_rate: float = 2e-5, 
                device: str = None, model_save_path: str = 'best_model.pth') -> nn.Module:
    """
    Train a model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use (cpu or cuda)
        model_save_path: Path to save the best model
        
    Returns:
        Trained model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Training on device: {device}")
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='weighted')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save training history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['val_f1'].append(val_f1)
        
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'epoch': epoch
            }, model_save_path)
            logger.info(f"Model saved to {model_save_path}")
    
    # Save training history
    with open(f"{model_save_path.split('.')[0]}_history.json", 'w') as f:
        json.dump(training_history, f)
    
    # Load best model
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1} with val_loss: {checkpoint['val_loss']:.4f}")
    
    return model


def train_segmentation_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                           epochs: int = 10, learning_rate: float = 2e-5, 
                           device: str = None, model_save_path: str = 'best_segmentation_model.pth') -> nn.Module:
    """
    Train a segmentation model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use (cpu or cuda)
        model_save_path: Path to save the best model
        
    Returns:
        Trained model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Training segmentation model on device: {device}")
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding (0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            segment_labels = batch['segment_labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Reshape for loss calculation
            # outputs: [batch_size, seq_len, num_classes]
            # segment_labels: [batch_size, seq_len]
            batch_size, seq_len, num_classes = outputs.shape
            loss = criterion(outputs.view(-1, num_classes), segment_labels.view(-1))
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_iou_sum = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                segment_labels = batch['segment_labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                
                # Loss calculation
                batch_size, seq_len, num_classes = outputs.shape
                loss = criterion(outputs.view(-1, num_classes), segment_labels.view(-1))
                val_loss += loss.item()
                
                # Calculate IoU for each sample
                preds = torch.argmax(outputs, dim=2)
                for i in range(batch_size):
                    # Only consider positions with valid labels (not padding)
                    mask = attention_mask[i].bool()
                    pred_i = preds[i, mask]
                    true_i = segment_labels[i, mask]
                    
                    # Calculate IoU for each class and average
                    class_iou = []
                    for cls in range(1, num_classes):  # Skip padding class
                        pred_cls = (pred_i == cls)
                        true_cls = (true_i == cls)
                        
                        if true_cls.sum() == 0 and pred_cls.sum() == 0:
                            # Class not present, skip
                            continue
                        
                        intersection = (pred_cls & true_cls).sum().float()
                        union = (pred_cls | true_cls).sum().float()
                        
                        if union > 0:
                            class_iou.append((intersection / union).item())
                    
                    if class_iou:
                        val_iou_sum += sum(class_iou) / len(class_iou)
                        val_samples += 1
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_iou = val_iou_sum / val_samples if val_samples > 0 else 0
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save training history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_iou'].append(val_iou)
        
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"Val IoU: {val_iou:.4f}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'epoch': epoch
            }, model_save_path)
            logger.info(f"Model saved to {model_save_path}")
    
    # Save training history
    with open(f"{model_save_path.split('.')[0]}_history.json", 'w') as f:
        json.dump(training_history, f)
    
    # Load best model
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1} with val_loss: {checkpoint['val_loss']:.4f}")
    
    return model


def predict_sequence(model: nn.Module, sequence: str, tokenizer, device: str = None) -> Dict[str, Any]:
    """
    Make prediction for a DNA sequence.
    
    Args:
        model: Trained model
        sequence: DNA sequence
        tokenizer: Tokenizer for the sequence
        device: Device to use (cpu or cuda)
        
    Returns:
        Dictionary with prediction results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model.to(device)
    
    # Tokenize the sequence
    encoding = tokenizer(
        sequence,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        if isinstance(model, SegmentNT):
            # For segmentation model, get predictions for each position
            preds = torch.argmax(outputs, dim=2).cpu().numpy()[0]
            attention_weights = None  # Not implemented for segmentation model
        else:
            # For classification model
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)
            pred_proba = probs[pred_class]
            attention_weights = model.get_attention_weights(input_ids, attention_mask)
            preds = pred_class
    
    return {
        'predictions': preds,
        'attention_weights': attention_weights,
        'input_ids': input_ids.cpu().numpy()[0] if not isinstance(model, SegmentNT) else None,
        'tokens': tokenizer.convert_ids_to_tokens(input_ids[0]) if not isinstance(model, SegmentNT) else None
    }

def visualize_attention(sequence: str, attention_weights: np.ndarray, tokens: List[str]):
    """
    Visualize attention weights.
    
    Args:
        sequence: Original DNA sequence
        attention_weights: Attention weights from model
        tokens: Tokenized sequence
    """
    # Get attention for [CLS] token (classification) or average attention
    if len(attention_weights.shape) == 3:
        # Take attention from last layer and average over heads
        attention = attention_weights[-1].mean(axis=0)[0]  # [CLS] token attention
    else:
        attention = attention_weights.mean(axis=0)[0]
    
    # Filter out special tokens and get corresponding positions in original sequence
    valid_indices = [i for i, token in enumerate(tokens) if token not in ['[CLS]', '[SEP]', '[PAD]']]
    valid_tokens = [tokens[i] for i in valid_indices]
    valid_attention = [attention[i] for i in valid_indices]
    
    # Create a mapping from k-mers to original sequence positions
    pos_mapping = []
    current_pos = 0
    for token in valid_tokens:
        token_len = len(token.replace("_", ""))  # Handle wordpiece tokens
        pos_mapping.append((current_pos, current_pos + token_len))
        current_pos += token_len
    
    # Create hover text showing k-mer and attention score
    hover_text = []
    for token, score in zip(valid_tokens, valid_attention):
        hover_text.append(f"Token: {token}<br>Attention: {score:.4f}")
    
    # Create figure
    fig = go.Figure()
    
    # Add attention scores as a bar chart
    fig.add_trace(go.Bar(
        x=[f"{i}" for i in range(len(valid_tokens))],
        y=valid_attention,
        hovertext=hover_text,
        hoverinfo="text",
        marker_color='royalblue'
    ))
    
    fig.update_layout(
        title="Attention Scores for Sequence Tokens",
        xaxis_title="Token Position",
        yaxis_title="Attention Score",
        hovermode="closest",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show sequence with attention highlights
    st.subheader("Sequence with Attention Highlights")
    
    # Create a color scale based on attention scores
    max_attention = max(valid_attention)
    min_attention = min(valid_attention)
    
    highlighted_seq = ""
    for (start, end), token, score in zip(pos_mapping, valid_tokens, valid_attention):
        # Normalize attention score for color
        norm_score = (score - min_attention) / (max_attention - min_attention)
        color = f"rgb({int(255 * norm_score)}, {int(255 * (1 - norm_score))}, 0)"
        highlighted_seq += f'<span style="background-color: {color}; border-radius: 3px; padding: 1px 2px;" title="Attention: {score:.4f}">{sequence[start:end]}</span>'
    
    st.markdown(highlighted_seq, unsafe_allow_html=True)

def visualize_segmentation(sequence: str, predictions: np.ndarray, class_names: List[str]):
    """
    Visualize segmentation predictions.
    
    Args:
        sequence: Original DNA sequence
        predictions: Predicted class for each position
        class_names: List of class names
    """
    # Create a color map for classes
    colors = px.colors.qualitative.Plotly
    if len(class_names) > len(colors):
        colors = colors * (len(class_names) // len(colors) + 1)
    
    # Create hover text
    hover_text = [f"Position: {i}<br>Base: {base}<br>Class: {class_names[pred]}" 
                 for i, (base, pred) in enumerate(zip(sequence, predictions[:len(sequence)]))]
    
    # Create figure
    fig = go.Figure()
    
    # Add sequence as text markers
    fig.add_trace(go.Scatter(
        x=list(range(len(sequence))),
        y=[0] * len(sequence),
        mode="text",
        text=list(sequence),
        textposition="middle center",
        textfont=dict(size=14),
        hovertext=hover_text,
        hoverinfo="text",
        showlegend=False
    ))
    
    # Add prediction background
    for cls in range(1, len(class_names)):  # Skip padding class
        mask = predictions[:len(sequence)] == cls
        if np.any(mask):
            fig.add_trace(go.Scatter(
                x=np.where(mask)[0],
                y=[0] * np.sum(mask),
                mode="markers",
                marker=dict(
                    color=colors[cls],
                    size=15,
                    opacity=0.3,
                    symbol="square"
                ),
                name=class_names[cls],
                hoverinfo="skip"
            ))
    
    fig.update_layout(
        title="Genomic Segmentation",
        xaxis_title="Position",
        yaxis=dict(visible=False),
        height=300,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="AgroNT - Genomic Sequence Analysis",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 AgroNT - Genomic Sequence Analysis")
    st.markdown("""
    This application provides tools for analyzing genomic sequences using transformer-based models.
    """)
    
    # Initialize data handler
    data_handler = GenomicDataHandler()
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["Nucleotide Transformer (Classification)", "AgroNT (Plant-Specific)", "SegmentNT (Segmentation)"]
    )
    
    task_type = "classification" if "Classification" in model_type else "segmentation"
    
    # Model parameters
    if task_type == "classification":
        num_classes = st.sidebar.number_input("Number of Classes", min_value=2, max_value=10, value=2)
    else:
        num_classes = st.sidebar.number_input("Number of Segment Classes", min_value=2, max_value=10, value=5)
        class_names = []
        for i in range(num_classes):
            class_names.append(st.sidebar.text_input(f"Class {i} Name", value=f"Class_{i}"))
    
    # Initialize model
    if model_type == "Nucleotide Transformer (Classification)":
        model = NucleotideTransformer(num_labels=num_classes)
    elif model_type == "AgroNT (Plant-Specific)":
        model = AgroNT(num_labels=num_classes)
    else:
        model = SegmentNT(num_classes=num_classes)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Sequence Analysis", "Train Model", "Data Exploration"])
    
    with tab1:
        st.header("Sequence Analysis")
        
        # Input options
        input_method = st.radio("Input Method", ["Direct Input", "Fetch from Genome", "Upload FASTA"])
        
        sequence = ""
        if input_method == "Direct Input":
            sequence = st.text_area("Enter DNA Sequence", height=150, max_chars=2000)
        elif input_method == "Fetch from Genome":
            col1, col2, col3, col4 = st.columns(4)
            species = col1.selectbox("Species", ["human", "mouse", "arabidopsis"])
            chromosome = col2.text_input("Chromosome", "1")
            start = col3.number_input("Start Position", min_value=1, value=1000000)
            end = col4.number_input("End Position", min_value=start+1, value=1000200)
            
            if st.button("Fetch Sequence"):
                try:
                    fasta = data_handler.fetch_genome_sequence(species, chromosome, start, end)
                    # Parse FASTA to get sequence
                    with StringIO(fasta) as handle:
                        for record in SeqIO.parse(handle, "fasta"):
                            sequence = str(record.seq)
                    st.success(f"Fetched {len(sequence)} bp sequence")
                except Exception as e:
                    st.error(f"Error fetching sequence: {e}")
        else:
            uploaded_file = st.file_uploader("Upload FASTA File", type=["fasta", "fa"])
            if uploaded_file:
                try:
                    sequences = data_handler.process_sequence_data(uploaded_file)
                    sequence = sequences[0]  # Take first sequence
                    st.success(f"Loaded sequence with {len(sequence)} bp")
                except Exception as e:
                    st.error(f"Error processing FASTA file: {e}")
        
        if sequence:
            st.subheader("Sequence Info")
            col1, col2 = st.columns(2)
            col1.metric("Sequence Length", f"{len(sequence)} bp")
            col2.metric("GC Content", f"{100 * (sequence.count('G') + sequence.count('C')) / len(sequence):.1f}%")
            
            # Show k-mer tokens if sequence is short enough
            if len(sequence) <= 100:
                k = st.slider("k-mer Size", min_value=3, max_value=8, value=6)
                kmers = data_handler.kmer_tokenize(sequence, k=k)
                st.write("k-mers:", ", ".join(kmers))
            
            # Make prediction
            if st.button("Analyze Sequence"):
                with st.spinner("Processing..."):
                    try:
                        result = predict_sequence(
                            model,
                            sequence,
                            model.tokenizer,
                            device='cuda' if torch.cuda.is_available() else 'cpu'
                        )
                        
                        if task_type == "classification":
                            st.subheader("Prediction Results")
                            st.write(f"Predicted class: {result['predictions']}")
                            
                            if result['attention_weights'] is not None:
                                st.subheader("Attention Visualization")
                                visualize_attention(
                                    sequence,
                                    result['attention_weights'],
                                    result['tokens']
                                )
                        else:
                            st.subheader("Segmentation Results")
                            visualize_segmentation(
                                sequence,
                                result['predictions'],
                                class_names
                            )
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
    
    with tab2:
        st.header("Train Model")
        st.warning("This feature requires pre-prepared training data.")
        
        # Training options
        train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"])
        if train_file:
            try:
                df = pd.read_csv(train_file)
                st.write("Preview:", df.head())
                
                # Select columns
                seq_col = st.selectbox("Sequence Column", df.columns)
                label_col = st.selectbox("Label Column", df.columns)
                
                # Split data
                test_size = st.slider("Validation Size", 0.1, 0.5, 0.2)
                random_state = st.number_input("Random State", value=42)
                
                if st.button("Train Model"):
                    sequences = df[seq_col].tolist()
                    labels = df[label_col].tolist()
                    
                    # Split data
                    train_seq, val_seq, train_labels, val_labels = train_test_split(
                        sequences, labels, test_size=test_size, random_state=random_state
                    )
                    
                    # Create datasets
                    if task_type == "classification":
                        train_dataset = GenomicDataset(train_seq, train_labels, model.tokenizer)
                        val_dataset = GenomicDataset(val_seq, val_labels, model.tokenizer)
                    else:
                        # For segmentation, assume labels are already lists
                        train_dataset = SegmentationDataset(train_seq, train_labels, model.tokenizer)
                        val_dataset = SegmentationDataset(val_seq, val_labels, model.tokenizer)
                    
                    # Create data loaders
                    batch_size = st.slider("Batch Size", 8, 64, 16)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size)
                    
                    # Training parameters
                    epochs = st.slider("Epochs", 1, 50, 10)
                    learning_rate = st.number_input("Learning Rate", value=2e-5)
                    
                    # Train model
                    with st.spinner("Training in progress..."):
                        if task_type == "classification":
                            trained_model = train_model(
                                model,
                                train_loader,
                                val_loader,
                                epochs=epochs,
                                learning_rate=learning_rate
                            )
                        else:
                            trained_model = train_segmentation_model(
                                model,
                                train_loader,
                                val_loader,
                                epochs=epochs,
                                learning_rate=learning_rate
                            )
                    
                    st.success("Training completed!")
                    model = trained_model  # Update the model with trained version
                    
            except Exception as e:
                st.error(f"Error during training: {e}")
    
    with tab3:
        st.header("Data Exploration")
        
        species = st.selectbox("Select Species", ["human", "mouse", "arabidopsis"])
        data_type = st.selectbox("Data Type", ["expression", "regulatory"])
        
        if st.button("Load Data"):
            try:
                if data_type == "expression":
                    df = data_handler.load_expression_data(species)
                else:
                    df = data_handler.load_regulatory_data(species)
                
                st.write(f"Data Shape: {df.shape}")
                st.write("Preview:", df.head())
                
                # Basic visualization
                if data_type == "expression":
                    gene_col = st.selectbox("Select Gene Column", df.columns)
                    value_col = st.selectbox("Select Value Column", [c for c in df.columns if c != gene_col])
                    
                    fig = px.histogram(df, x=value_col, nbins=50, title="Expression Value Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Regulatory data
                    chrom_col = st.selectbox("Select Chromosome Column", df.columns)
                    start_col = st.selectbox("Select Start Position Column", df.columns)
                    
                    fig = px.histogram(df, x=chrom_col, title="Regulatory Elements by Chromosome")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading data: {e}")

if __name__ == "__main__":
    main()