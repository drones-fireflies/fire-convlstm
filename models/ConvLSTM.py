import torch
import torch.nn as nn

# =============
# ConvLSTM Cell
# =============
class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell"""
    def __init__(self, input_channels, hidden_size, kernel_size):
        """
        Args:
            input_channels (int): Number of input channels.
            hidden_size (int): Number of hidden channels.
            kernel_size (int): Size of the convolutional kernel.
        """
        super().__init__()
        padding = kernel_size // 2
        self.hidden_size = hidden_size

        # Convolution layer that computes all four gates
        self.conv = nn.Conv2d(
            input_channels + hidden_size,     # Concatenate input and hidden states
            4 * hidden_size,                  # Four gates: i, f, o, g
            kernel_size,
            padding=padding
        )

    def forward(self, x_t, hidden_state):

        h_t, c_t = hidden_state

        # Combine input and previous hidden state
        combined = torch.cat([x_t, h_t], dim=1)

        # Compute gates
        gates = self.conv(combined)
        i_gate, f_gate, o_gate, g_gate = torch.split(gates, self.hidden_size, dim=1)

        # Apply activations
        i_gate = torch.sigmoid(i_gate)    # Input gate
        f_gate = torch.sigmoid(f_gate)    # Forget gate
        o_gate = torch.sigmoid(o_gate)    # Output gate
        g_gate = torch.tanh(g_gate)       # Candidate cell state

        # Compute next cell and hidden states
        c_next = f_gate * c_t + i_gate * g_gate
        h_next = o_gate * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (torch.zeros(batch_size, self.hidden_size, height, width, device=device),
                torch.zeros(batch_size, self.hidden_size, height, width, device=device))

# ==============
# ConvLSTM Model
# ==============
class ConvLSTM(nn.Module):
    """Configurable ConvLSTM model using parameters from YAML."""
    def __init__(self, input_channels, config):
        """
        Args:
            input_channels (int): Number of input channels.
            config (dict):
                - num_lstm_layers (int)
                - hidden_size (int)
                - use_sigmoid (bool)
                - output_size (int)
                - convlstm_kernel_size (int)
                - decoder_conv_kernel_size (int)
        """

        super().__init__()
        self.num_layers = config["num_lstm_layers"]
        self.hidden_size = config["hidden_size"]
        self.use_sigmoid = config["use_sigmoid"]
        self.output_size = config["output_size"]

        # Build stacked ConvLSTM cells
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                input_channels if i == 0 else self.hidden_size,
                self.hidden_size,
                config["convlstm_kernel_size"]
            )
            for i in range(self.num_layers)
        ])

        # Decode layer
        self.decoder = nn.Conv2d(
            self.hidden_size,
            self.output_size,
            kernel_size=config["decoder_conv_kernel_size"]
        )

    def forward(self, x):

        batch_size, seq_len, _, height, width = x.size()

        # Initialize hidden and cell states for each layer
        hidden_states = []
        cell_states = []
        for layer in range(self.num_layers):
            h_0, c_0 = self.cells[layer].init_hidden(batch_size, height, width, x.device)
            hidden_states.append(h_0)
            cell_states.append(c_0)

        # Process input sequence over time
        for t in range(seq_len):
            input_t = x[:, t]
            next_hidden_states = []
            next_cell_states = []

            # Pass through each ConvLSTM layer
            for layer in range(self.num_layers):
                h_next, c_next = self.cells[layer](input_t, (hidden_states[layer], cell_states[layer]))
                next_hidden_states.append(h_next)
                next_cell_states.append(c_next)
                input_t = h_next

            hidden_states, cell_states = next_hidden_states, next_cell_states

        # Decode the last hidden state
        output = self.decoder(hidden_states[-1])

        # Optional sigmoid activation on output
        return torch.sigmoid(output) if self.use_sigmoid else output