import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # No longer needed for direct plotting
import networkx as nx
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import torch_geometric.nn as geo_nn
from torch_geometric.data import Data
# import plotly.graph_objects as go # No longer needed
# import plotly.express as px # No longer needed
# from plotly.subplots import make_subplots # No longer needed
# import plotly.offline as pyo # No longer needed
import warnings
import json # Needed for JSON export

warnings.filterwarnings('ignore')

# --- Keep LSTMModel and GCNModel definitions as they are ---
class LSTMModel(nn.Module):
    # ... (previous code) ...
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GCNModel(torch.nn.Module):
    # ... (previous code) ...
    def __init__(self, num_node_features, hidden_channels):
        super(GCNModel, self).__init__()
        self.conv1 = geo_nn.GCNConv(num_node_features, hidden_channels)
        self.conv2 = geo_nn.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = geo_nn.GCNConv(hidden_channels, 1) # Predicts a single value (e.g., edge weight)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.5, train=self.training)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        return x


# --- Keep preprocess_single_file, extract_flow_events, infer_person_movements ---
# --- update_flow_graph, prepare_timeseries_data_from_file, train_lstm_incrementally ---
# --- prepare_graph_data_from_movements, train_gcn_incrementally ---
# --- (all these helper functions remain the same as before) ---
def preprocess_single_file(file_path):
    """
    处理单个数据文件
    """
    # 根据文件扩展名加载数据
    if file_path.endswith('.csv'):
        # 尝试不同的编码格式
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
        for encoding in encodings:
            try:
                # print(f"尝试使用 {encoding} 编码读取文件...") # Reduced verbosity
                df = pd.read_csv(file_path, encoding=encoding)
                # print(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                # print(f"{encoding} 编码失败，尝试下一个编码...")
                if encoding == encodings[-1]:  # 如果是最后一个编码仍然失败
                    print(f"无法读取文件 {file_path}，所有编码尝试均失败")
                    return None
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {str(e)}")
                return None
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"读取Excel文件 {file_path} 时出错: {str(e)}")
            return None
    else:
        print(f"不支持的文件类型: {file_path}")
        return None

    # print(f"文件 {file_path} 包含 {df.shape[0]} 行和 {df.shape[1]} 列")
    # print(f"列名: {df.columns.tolist()}")

    # 动态映射列名 - 查找包含关键字的列
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'ts' in col_lower or 'time' in col_lower:
            col_map['timestamp'] = col
        elif 'device' in col_lower or 'sn' in col_lower:
            col_map['device_id'] = col
        elif 'region' in col_lower or 'location' in col_lower or '区域' in col: # Added 区域
             col_map['location_id'] = col
        elif '红外' in col or 'ir' in col_lower or '感应' in col: # Added 感应
             col_map['ir_status'] = col

    # 检查必要列是否都映射成功
    required_keys = ['timestamp', 'location_id', 'ir_status']
    mapped_keys = list(col_map.keys())
    if not all(key in mapped_keys for key in required_keys):
         missing = [key for key in required_keys if key not in mapped_keys]
         print(f"文件 {file_path} 缺少必要列的映射: {missing}. 可用列: {df.columns.tolist()}")
         # Try default mapping as fallback
         default_mapping = {
             'ts': 'timestamp',
             'device_sn': 'device_id',
             'region': 'location_id',
             '人体红外感应;1-有人；0-无人': 'ir_status'
         }
         found_default = False
         for default_col, target_key in default_mapping.items():
             if default_col in df.columns and target_key not in col_map:
                 col_map[target_key] = default_col
                 found_default = True
         if not all(key in col_map for key in required_keys):
              print("尝试默认映射后仍然缺少必要列，跳过文件。")
              return None
         else:
              print("使用了部分默认列名映射。")


    # 创建映射后的DataFrame
    new_df = pd.DataFrame()
    for target_col, source_col in col_map.items():
        if source_col in df.columns:
            new_df[target_col] = df[source_col]
        else:
            print(f"警告：源列 '{source_col}' (映射到 '{target_col}') 在文件中未找到。")


    # 处理时间戳
    if 'timestamp' not in new_df.columns:
        print("错误：处理后无时间戳列。")
        return None

    try:
        sample_ts = new_df['timestamp'].iloc[0]
        # print(f"时间戳样例: {sample_ts}, 类型: {type(sample_ts)}") # Reduced verbosity

        # If numeric (Unix timestamp)
        if pd.api.types.is_numeric_dtype(new_df['timestamp']):
             # Check magnitude for ms vs s
             if new_df['timestamp'].max() > 2 * 10**10: # Likely milliseconds
                 new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms', errors='coerce')
             else: # Likely seconds
                 new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='s', errors='coerce')
        # If string
        elif pd.api.types.is_string_dtype(new_df['timestamp']):
             new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], errors='coerce')
        # If already datetime (from Excel)
        elif pd.api.types.is_datetime64_any_dtype(new_df['timestamp']):
             pass # Already correct type
        else:
             print(f"未知的时间戳格式: {type(sample_ts)}")
             new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], errors='coerce')

        # Drop rows where timestamp conversion failed
        original_rows = len(new_df)
        new_df.dropna(subset=['timestamp'], inplace=True)
        if len(new_df) < original_rows:
            print(f"警告：移除了 {original_rows - len(new_df)} 行无效时间戳的记录")

        if new_df.empty:
            print("错误：处理时间戳后数据为空。")
            return None

    except Exception as e:
        print(f"转换时间戳时发生严重错误: {str(e)}")
        return None # Skip file if timestamp conversion fails critically

    # 处理红外感应状态
    if 'ir_status' not in new_df.columns:
         print("错误：处理后无红外状态列。")
         return None

    try:
        # Common pattern: "1-有人", "0-无人"
        if pd.api.types.is_string_dtype(new_df['ir_status']):
            new_df['ir_status'] = new_df['ir_status'].str.extract(r'^(\d+)').astype(int)
        elif pd.api.types.is_numeric_dtype(new_df['ir_status']):
            new_df['ir_status'] = new_df['ir_status'].astype(int)
        else:
             # Fallback: try converting directly, coercing errors
             new_df['ir_status'] = pd.to_numeric(new_df['ir_status'], errors='coerce')
             new_df.dropna(subset=['ir_status'], inplace=True) # Remove rows where conversion failed
             if not new_df.empty:
                  new_df['ir_status'] = new_df['ir_status'].astype(int)


        # Ensure status is 0 or 1 after processing
        valid_status = new_df['ir_status'].isin([0, 1])
        if not valid_status.all():
            print(f"警告: 文件 {file_path} 包含无效的红外状态值 (非 0 或 1).")
            # Option: Keep only valid rows, or attempt further cleaning
            new_df = new_df[valid_status] # Keep only valid 0/1 rows


        if new_df.empty:
            print("错误：处理红外状态后数据为空。")
            return None

    except Exception as e:
         print(f"处理红外状态时出错: {str(e)}")
         return None # Skip file if IR status processing fails


    # 处理位置ID
    if 'location_id' not in new_df.columns:
        print("错误：处理后无位置ID列。")
        return None
    new_df['location_id'] = new_df['location_id'].astype(str).str.strip() # Ensure string and remove whitespace

    # Select and sort final columns
    final_cols = ['timestamp', 'location_id', 'ir_status']
    if 'device_id' in new_df.columns:
         final_cols.append('device_id')
         new_df['device_id'] = new_df['device_id'].astype(str).str.strip()

    new_df = new_df[final_cols].sort_values('timestamp').reset_index(drop=True)

    # Display processed info (reduced verbosity)
    # print(f"处理后: {new_df.shape[0]} 行, {new_df.shape[1]} 列.")
    # if not new_df.empty:
    #     print(f"  时间范围: {new_df['timestamp'].min()} -> {new_df['timestamp'].max()}")
    #     print(f"  位置 ({new_df['location_id'].nunique()}): {sorted(new_df['location_id'].unique())[:10]}...") # Show only first 10
    #     print(f"  红外状态值: {new_df['ir_status'].unique()}")

    return new_df

def extract_flow_events(data):
    """
    根据红外传感器状态变化推断人员流动事件
    """
    if not all(col in data.columns for col in ['timestamp', 'location_id', 'ir_status']):
        print("数据缺少必要列 ['timestamp', 'location_id', 'ir_status']")
        return pd.DataFrame(columns=['timestamp', 'location', 'event_type'])

    flow_data = data[['timestamp', 'location_id', 'ir_status']].copy()
    flow_events = []

    # Use shift within groupby for efficiency
    flow_data.sort_values(['location_id', 'timestamp'], inplace=True)
    flow_data['prev_status'] = flow_data.groupby('location_id')['ir_status'].shift(1)
    # Fill NaN for the first entry of each group. Assume initial state is 0 if first reading is 1.
    flow_data['prev_status'] = flow_data.apply(
        lambda row: 0 if pd.isna(row['prev_status']) and row['ir_status'] == 1 else row['prev_status'],
        axis=1
    )
    flow_data['prev_status'].fillna(0, inplace=True) # Fill remaining NaNs (e.g., if first reading is 0)
    flow_data['prev_status'] = flow_data['prev_status'].astype(int)

    # Calculate status change: 1 = entry (0->1), -1 = exit (1->0)
    flow_data['status_change'] = flow_data['ir_status'] - flow_data['prev_status']

    # Extract entry events (status_change == 1)
    entries = flow_data[flow_data['status_change'] == 1][['timestamp', 'location_id']]
    entries['event_type'] = 'entry'
    entries.rename(columns={'location_id': 'location'}, inplace=True)

    # Extract exit events (status_change == -1)
    exits = flow_data[flow_data['status_change'] == -1][['timestamp', 'location_id']]
    exits['event_type'] = 'exit'
    exits.rename(columns={'location_id': 'location'}, inplace=True)

    # Combine and sort
    flow_df = pd.concat([entries, exits], ignore_index=True)
    if not flow_df.empty:
        flow_df = flow_df.sort_values('timestamp').reset_index(drop=True)

    return flow_df

def infer_person_movements(flow_events, time_threshold_seconds=300):
    """
    根据入口和出口事件推断人员移动
    Looks for an 'entry' event within 'time_threshold_seconds' after an 'exit' event at a different location.
    """
    if flow_events.empty or not all(col in flow_events.columns for col in ['timestamp', 'location', 'event_type']):
        return pd.DataFrame(columns=['from', 'to', 'timestamp', 'arrival_time', 'time_diff'])

    movements = []
    exits = flow_events[flow_events['event_type'] == 'exit'].copy()
    entries = flow_events[flow_events['event_type'] == 'entry'].copy()

    if exits.empty or entries.empty:
        return pd.DataFrame(columns=['from', 'to', 'timestamp', 'arrival_time', 'time_diff'])

    # Sort for efficient searching
    exits.sort_values('timestamp', inplace=True)
    entries.sort_values('timestamp', inplace=True)

    # Use searchsorted for potentially faster lookups (if data is large)
    entry_timestamps = entries['timestamp'].values
    entry_locations = entries['location'].values

    time_threshold = pd.Timedelta(seconds=time_threshold_seconds)

    for _, exit_event in exits.iterrows():
        exit_time = exit_event['timestamp']
        exit_location = exit_event['location']
        max_arrival_time = exit_time + time_threshold

        # Find potential entries within the time window
        start_idx = np.searchsorted(entry_timestamps, exit_time, side='right')
        end_idx = np.searchsorted(entry_timestamps, max_arrival_time, side='right')

        potential_indices = []
        for idx in range(start_idx, end_idx):
            # Check location difference
            if entry_locations[idx] != exit_location:
                 potential_indices.append(idx)

        if potential_indices:
            # Find the index of the *earliest* valid entry within the window
            best_entry_idx = potential_indices[0] # Since entries are sorted by time

            nearest_entry = entries.iloc[best_entry_idx]

            movements.append({
                'from': exit_location,
                'to': nearest_entry['location'],
                'timestamp': exit_time, # Departure time
                'arrival_time': nearest_entry['timestamp'], # Arrival time
                'time_diff': (nearest_entry['timestamp'] - exit_time).total_seconds()
            })
            # Optional: Mark the entry as 'used' to prevent matching multiple exits?
            # This depends on whether one entry can explain multiple preceding exits.
            # For simplicity, we allow it for now.

    if movements:
        movement_df = pd.DataFrame(movements)
        movement_df.sort_values('timestamp', inplace=True)
        return movement_df
    else:
        return pd.DataFrame(columns=['from', 'to', 'timestamp', 'arrival_time', 'time_diff'])

def update_flow_graph(G, movements):
    """
    根据新的移动数据更新流动图 (NetworkX DiGraph)
    """
    if movements.empty:
        return G

    # Add nodes if they don't exist
    all_locations = set(movements['from']).union(set(movements['to']))
    for loc in all_locations:
        if not G.has_node(loc):
            G.add_node(loc) # Add node attributes later if needed

    # Update edge weights
    for _, move in movements.iterrows():
        u, v = move['from'], move['to']
        if G.has_edge(u, v):
            G[u][v]['weight'] = G[u][v].get('weight', 0) + 1
            # Update timestamp list for the edge
            G[u][v]['timestamps'] = G[u][v].get('timestamps', []) + [move['timestamp']]
        else:
            G.add_edge(u, v, weight=1, timestamps=[move['timestamp']]) # Start weight at 1

    return G

def prepare_timeseries_data_from_file(data, all_locations, seq_length=10):
    """
    从单个文件准备LSTM的时间序列数据
    Input: DataFrame with 'timestamp', 'location_id', 'ir_status'
    Output: Tensors X (sequences), y (targets)
    """
    if data.empty or not all(col in data.columns for col in ['timestamp', 'location_id', 'ir_status']):
        # print("LSTM Data Prep: Missing required columns or empty data.")
        return None, None

    locations = sorted(list(all_locations))
    location_map = {loc: i for i, loc in enumerate(locations)}
    num_locations = len(locations)

    # Pivot data to have locations as columns, timestamp as index
    try:
        # Use pivot_table to handle duplicates (take max status if multiple readings at same time)
        status_pivot = pd.pivot_table(data, values='ir_status', index='timestamp', columns='location_id', aggfunc='max')

        # Reindex to ensure all locations are present and in correct order, fill missing with 0
        status_pivot = status_pivot.reindex(columns=locations, fill_value=0)

        # Upsample or resample to a fixed frequency if needed?
        # For now, use the unique timestamps present in the data.
        # If timestamps are very sparse, this might be problematic.
        # Consider resampling: status_pivot = status_pivot.resample('1T').max().fillna(0) # Resample to 1 minute

    except Exception as e:
        print(f"Error creating status pivot table: {e}")
        return None, None

    status_matrix = status_pivot.values # Numpy array

    if len(status_matrix) < seq_length + 1:
        # print(f"LSTM Data Prep: Not enough timestamps ({len(status_matrix)}) for sequence length {seq_length}.")
        return None, None

    # Create sequences
    X, y = [], []
    for i in range(len(status_matrix) - seq_length):
        X.append(status_matrix[i : i + seq_length])
        y.append(status_matrix[i + seq_length])

    if not X:
        return None, None

    X = np.array(X)
    y = np.array(y)

    # Normalize data? LSTM might benefit from MinMaxScaler per feature (location)
    # scaler = MinMaxScaler()
    # X_shape = X.shape
    # X = scaler.fit_transform(X.reshape(-1, X_shape[-1])).reshape(X_shape)
    # y = scaler.transform(y) # Use the same scaler

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    return X_tensor, y_tensor

def train_lstm_incrementally(model, optimizer, criterion, X, y, batch_size=32, epochs=5, device='cpu'):
    """
    增量训练LSTM模型
    """
    if X is None or y is None or len(X) == 0:
        print("Skipping LSTM training: No data provided.")
        return model

    # Move model to device
    model.to(device)

    # Create data loader
    dataset = TensorDataset(X.to(device), y.to(device)) # Move data to device
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    print(f"Starting LSTM incremental training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        for batch_X, batch_y in dataloader:
            # Data already on device from DataLoader
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Epoch {epoch+1}, Batch {batch_count+1}: NaN loss detected. Skipping batch.")
                continue

            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            print(f'  Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f}')
        else:
            print(f"  Epoch [{epoch+1}/{epochs}], No batches processed (possibly all NaN losses).")
    print("LSTM training epoch finished.")
    return model

def prepare_graph_data_from_movements(G, all_locations, use_edge_weight=True):
    """
    从更新后的 NetworkX 图 G 准备 PyTorch Geometric Data 对象
    Node features: [total_out_weight, total_in_weight] (normalized)
    Edge attributes: edge weight (optional)
    """
    if not G or G.number_of_nodes() == 0:
        print("GCN Data Prep: Graph is empty.")
        return None

    locations = sorted(list(all_locations))
    location_map = {loc: i for i, loc in enumerate(locations)}
    num_nodes = len(locations)
    node_features = np.zeros((num_nodes, 2), dtype=np.float32)

    # Calculate node features (total out/in weight)
    for i, loc in enumerate(locations):
        if G.has_node(loc):
            out_weight = sum(G[loc][succ].get('weight', 0) for succ in G.successors(loc))
            in_weight = sum(G[pred][loc].get('weight', 0) for pred in G.predecessors(loc))
            node_features[i, 0] = out_weight
            node_features[i, 1] = in_weight

    # Normalize node features
    scaler = MinMaxScaler()
    # Check if features are not all zero before scaling
    if np.any(node_features):
         # Scale each feature column independently
         node_features = scaler.fit_transform(node_features)
    else:
         print("GCN Data Prep: Node features are all zero.")
         # Keep features as zeros, GCN might still learn from topology

    x = torch.tensor(node_features, dtype=torch.float)

    # Prepare edge index
    edge_list = []
    edge_weights = []
    valid_edges = 0
    for u, v, data in G.edges(data=True):
        if u in location_map and v in location_map:
            u_idx, v_idx = location_map[u], location_map[v]
            edge_list.append([u_idx, v_idx])
            edge_weights.append(data.get('weight', 1.0)) # Default weight 1 if missing
            valid_edges += 1
        # else:
            # print(f"Warning: Edge ({u}, {v}) involves unknown location.")

    if valid_edges == 0:
        print("GCN Data Prep: No valid edges found between known locations.")
        # Return data with features but no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float) if use_edge_weight else None

    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        if use_edge_weight:
             # Normalize edge weights?
             weights_array = np.array(edge_weights).reshape(-1, 1)
             if np.any(weights_array): # Avoid scaling if all zeros
                 edge_scaler = MinMaxScaler()
                 weights_array = edge_scaler.fit_transform(weights_array)
             edge_attr = torch.tensor(weights_array, dtype=torch.float)
             # Ensure edge_attr has shape [num_edges, 1] if GCN expects it
             if edge_attr.dim() == 1:
                  edge_attr = edge_attr.unsqueeze(1)

        else:
             edge_attr = None


    # Create PyG Data object
    graph_data = Data(x=x, edge_index=edge_index)
    if edge_attr is not None:
         graph_data.edge_attr = edge_attr

    return graph_data

def train_gcn_incrementally(model, optimizer, criterion, graph_data, epochs=5, device='cpu'):
    """
    增量训练GCN模型.
    Assumes model predicts node features or edge properties.
    Here, we assume it predicts node properties (like centrality or future state),
    so the target 'y' needs to be defined. If it predicts edge weights, criterion needs adjustment.

    Let's adapt it to predict edge weights (as the previous GCN model output dim was 1).
    """
    if graph_data is None or graph_data.num_edges == 0:
        print("Skipping GCN training: No graph data or no edges.")
        return model

    # Move model and data to device
    model.to(device)
    graph_data = graph_data.to(device)

    # Ensure edge_attr exists for training target
    if graph_data.edge_attr is None:
        print("Skipping GCN training: No edge attributes (weights) provided as target.")
        return model

    # Target: Normalized edge weights
    y_true = graph_data.edge_attr

    model.train()
    print(f"Starting GCN incremental training for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        # GCN typically outputs node embeddings. How to get edge predictions?
        # Option 1: Concatenate node embeddings and pass through a linear layer.
        # Option 2: Use an edge-based GNN layer.
        # Option 3: (Simpler) Assume the node output relates to edge weight (less accurate).
        # Let's try Option 3 based on the original GCN structure:
        # Output node embeddings
        node_embeddings = model(graph_data.x, graph_data.edge_index) # Shape [num_nodes, 1]

        # Predict edge weight based on source node's embedding (simplistic)
        source_nodes = graph_data.edge_index[0]
        y_pred = node_embeddings[source_nodes] # Shape [num_edges, 1]

        # Ensure shapes match for loss calculation
        if y_pred.shape != y_true.shape:
             print(f"Warning: GCN output shape {y_pred.shape} mismatch target shape {y_true.shape}. Reshaping target.")
             try:
                 y_true = y_true.view(y_pred.shape)
             except RuntimeError as e:
                 print(f"Error reshaping target tensor: {e}. Skipping GCN training epoch.")
                 continue # Skip this epoch if reshape fails

        loss = criterion(y_pred, y_true)

        if torch.isnan(loss):
             print(f"Epoch {epoch+1}: NaN loss detected in GCN training. Skipping update.")
             continue

        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f'  Epoch [{epoch+1}/{epochs}], GCN Loss: {loss.item():.6f}')
    print("GCN training epoch finished.")
    return model


# --- NEW Function to export data for JavaScript ---
def export_flow_data_for_js(all_movements, all_locations, time_bin_minutes=60, output_file='flow_data.json'):
    """
    Aggregates movement data into time bins and exports as JSON for the JS visualization.
    """
    if all_movements.empty:
        print("No movement data to export.")
        return

    print(f"\nExporting flow data for JavaScript visualization...")
    print(f"Aggregating data into {time_bin_minutes}-minute bins...")

    # Ensure timestamp is datetime
    all_movements['timestamp'] = pd.to_datetime(all_movements['timestamp'])

    # Define time bins based on departure time ('timestamp')
    bin_freq = f'{time_bin_minutes}T'
    all_movements['time_bin'] = all_movements['timestamp'].dt.floor(bin_freq)

    # Group by time bin, source, and destination
    flow_counts = all_movements.groupby(['time_bin', 'from', 'to']).size().reset_index(name='volume')

    # Get overall time range
    start_time = all_movements['timestamp'].min()
    end_time = all_movements['timestamp'].max()

    # Structure the data for JSON
    time_bins_data = []
    unique_bins = sorted(flow_counts['time_bin'].unique())

    print(f"Found {len(unique_bins)} unique time bins.")

    for bin_ts in unique_bins:
        bin_flows_df = flow_counts[flow_counts['time_bin'] == bin_ts]
        flows_list = []
        for _, row in bin_flows_df.iterrows():
            flows_list.append({
                'from': row['from'],
                'to': row['to'],
                'volume': int(row['volume']) # Ensure integer volume
            })

        # Format timestamp as ISO string (UTC assumed if no tz info)
        bin_ts_str = pd.Timestamp(bin_ts).isoformat() + "Z" # Add Z for UTC indication

        time_bins_data.append({
            'timestamp': bin_ts_str,
            'flows': flows_list
        })

    # Final JSON structure
    output_data = {
        'locations': sorted(list(all_locations)), # List of unique location IDs
        'startTime': start_time.isoformat() + "Z",
        'endTime': end_time.isoformat() + "Z",
        'timeBinMinutes': time_bin_minutes,
        'time_bins': time_bins_data
    }

    # Write to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2) # indent for readability
        print(f"Flow data successfully exported to {output_file}")
    except Exception as e:
        print(f"Error writing JSON file {output_file}: {e}")

def main():
    # Set data folder paths
    train_folder = r"E:\chesk\Desktop\应用统计建模大赛\训练集"
    test_folder = r"E:\chesk\Desktop\应用统计建模大赛\测试集"
    output_json_file = 'flow_data.json' # Output file for JS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
    print(f"Using device: {device}")


    # Initialization
    all_data_raw = [] # Store raw preprocessed data for unified LSTM/GCN training later if needed
    all_locations = set()
    processed_files_count = 0

    # --- Pass 1: Collect all locations and basic stats ---
    print("--- Pass 1: Scanning files for locations and time range ---")
    all_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder)] + \
                [os.path.join(test_folder, f) for f in os.listdir(test_folder)]

    min_global_time = pd.Timestamp.max
    max_global_time = pd.Timestamp.min

    for file_path in all_files:
         # print(f"Scanning file: {os.path.basename(file_path)}") # Verbose
         if not (file_path.endswith('.csv') or file_path.endswith('.xlsx') or file_path.endswith('.xls')):
             print(f"Skipping non-data file: {os.path.basename(file_path)}")
             continue

         data = preprocess_single_file(file_path)
         if data is not None and not data.empty:
             current_locations = data['location_id'].unique()
             all_locations.update(current_locations)
             all_data_raw.append(data) # Store preprocessed data
             min_global_time = min(min_global_time, data['timestamp'].min())
             max_global_time = max(max_global_time, data['timestamp'].max())
             processed_files_count += 1
         else:
             print(f"Failed to process or empty data in: {os.path.basename(file_path)}")


    if not all_locations:
        print("\nError: No valid location data found in any files. Exiting.")
        return
    if processed_files_count == 0:
        print("\nError: No files were processed successfully. Exiting.")
        return

    all_locations = sorted(list(all_locations)) # Ensure consistent order
    print(f"\n--- Scan Complete ---")
    print(f"Successfully scanned {processed_files_count} files.")
    print(f"Found {len(all_locations)} unique locations: {all_locations[:15]}...") # Show first 15
    print(f"Overall time range: {min_global_time} to {max_global_time}")


    # --- Process Data and Infer Movements ---
    print("\n--- Pass 2: Processing data and inferring movements ---")
    all_movements = pd.DataFrame()
    G = nx.DiGraph() # Initialize Flow Graph

    for i, data in enumerate(all_data_raw):
        print(f"Processing data chunk {i+1}/{len(all_data_raw)}...")
        # Extract flow events (entry/exit)
        flow_events = extract_flow_events(data)
        # Infer movements between locations
        movements = infer_person_movements(flow_events, time_threshold_seconds=600) # 10 min threshold

        if not movements.empty:
            all_movements = pd.concat([all_movements, movements], ignore_index=True)
            # Update the cumulative flow graph
            G = update_flow_graph(G, movements)

    if all_movements.empty:
         print("\nWarning: No movements were inferred from the data.")
         # Optionally, still proceed to export empty structure if needed by JS
    else:
         print(f"\nTotal inferred movements: {len(all_movements)}")
         print(f"Flow Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
         # print(f"Sample movements:\n{all_movements.head()}")


    # --- Initialize Models ---
    input_dim = len(all_locations)
    hidden_dim = 64
    num_layers = 2
    output_dim = input_dim # LSTM predicts status for all locations

    lstm_model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    lstm_criterion = nn.MSELoss()

    # GCN Features: Out-degree, In-degree
    gcn_model = GCNModel(num_node_features=2, hidden_channels=16).to(device)
    gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01)
    gcn_criterion = torch.nn.MSELoss() # Loss for edge weight prediction


    # --- Train Models Incrementally (using stored raw data chunks) ---
    print("\n--- Pass 3: Training models ---")
    for i, data in enumerate(all_data_raw):
         print(f"\nTraining on data chunk {i+1}/{len(all_data_raw)}...")
         # Prepare and Train LSTM
         X_lstm, y_lstm = prepare_timeseries_data_from_file(data, all_locations, seq_length=12) # Use slightly longer seq
         if X_lstm is not None and y_lstm is not None:
             lstm_model = train_lstm_incrementally(
                 lstm_model, lstm_optimizer, lstm_criterion, X_lstm, y_lstm,
                 batch_size=64, epochs=3, device=device
             )
         else:
             print("  Skipping LSTM training for this chunk (insufficient data).")


    # Prepare Graph Data ONCE using the FINAL graph G
    print("\nPreparing final graph data for GCN...")
    graph_data = prepare_graph_data_from_movements(G, all_locations, use_edge_weight=True)

    # Train GCN ONCE on the final graph structure
    if graph_data is not None:
         print("Training GCN model on the final aggregated graph...")
         gcn_model = train_gcn_incrementally(
             gcn_model, gcn_optimizer, gcn_criterion, graph_data, epochs=50, device=device # More epochs for GCN
         )
    else:
         print("  Skipping GCN training (no valid graph data).")


    # --- Save Models ---
    try:
        torch.save(lstm_model.state_dict(), 'lstm_model_final.pth')
        torch.save(gcn_model.state_dict(), 'gcn_model_final.pth')
        print("\nModels saved successfully.")
    except Exception as e:
        print(f"\nError saving models: {e}")


    # --- Export Data for Visualization ---
    # Use all inferred movements (from both train/test originally, now combined)
    if not all_movements.empty:
        export_flow_data_for_js(all_movements, all_locations, time_bin_minutes=60, output_file=output_json_file)
    else:
        # Export empty structure if needed by JS? Or just print warning.
        print("\nNo movement data to export for visualization.")
        # Create an empty JSON structure if required
        empty_output = {
            'locations': all_locations,
            'startTime': min_global_time.isoformat() + "Z" if min_global_time != pd.Timestamp.max else None,
            'endTime': max_global_time.isoformat() + "Z" if max_global_time != pd.Timestamp.min else None,
            'timeBinMinutes': 60,
            'time_bins': []
        }
        try:
             with open(output_json_file, 'w', encoding='utf-8') as f:
                 json.dump(empty_output, f, ensure_ascii=False, indent=2)
             print(f"Empty flow data structure exported to {output_json_file}")
        except Exception as e:
             print(f"Error writing empty JSON file {output_json_file}: {e}")


    print("\n--- Main script execution finished ---")

if __name__ == "__main__":
    main()
