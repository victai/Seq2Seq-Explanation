import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=2, bidirectional=False):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)
        if bidirectional == True:
            self.fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        hidden = self.dropout(hidden)
        if self.bidirectional == True:
            hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
            hidden = self.fc(torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2))
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers=2):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=n_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        #hidden = torch.sum(hidden, dim=0).unsqueeze(0)
        output = self.softmax(self.output_layer(output))
        
        return output, hidden


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, n_layers=1, dropout_p=0.1, bidirectional=False):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        if bidirectional == True:
            self.attn = nn.Linear(self.hidden_size*3, hidden_size)
        else:
            self.attn = nn.Linear(self.hidden_size*2, hidden_size)
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)
        if bidirectional == True:
            self.gru = nn.GRU(hidden_size*3, hidden_size, n_layers)
        else:
            self.gru = nn.GRU(hidden_size*2, hidden_size, n_layers)
        self.output_layer = nn.Linear(hidden_size*3, output_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        batch_size = encoder_outputs.shape[1]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((encoder_outputs, hidden[0].unsqueeze(1).repeat(1, self.max_length, 1)), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = F.softmax(torch.bmm(v, energy), dim=2)
        weighted = torch.bmm(attention, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)

        #output = F.relu(output)
        output, hidden = self.gru(rnn_input, hidden)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        embedded = embedded.squeeze(0)
        output = self.output_layer(torch.cat((output, weighted, embedded), dim=1))

        return output, hidden, attention


class Attn(nn.Module):
    def __init__(self, method, hidden_size, use_cuda=True):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden [1, 64, 512], encoder_outputs [14, 64, 512]
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = torch.zeros(batch_size, max_len) # B x S
        if self.use_cuda:
            attn_energies = attn_energies.cuda()

        if self.method == 'general':
            encoder_outputs = self.attn(encoder_outputs)
        attn_energies = torch.bmm(encoder_outputs.transpose(0, 1), hidden.transpose(0, 1).transpose(1, 2)).squeeze(-1)

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        # hidden [1, 512], encoder_output [1, 512]
        if self.method == 'dot':
            energy = hidden.squeeze(0).dot(encoder_output.squeeze(0))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.squeeze(0).dot(energy.squeeze(0))
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.squeeze(0).dot(energy.squeeze(0))
            return energy


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1, use_cuda=True):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size, use_cuda)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded) #[1, 64, 512]
        if(embedded.size(0) != 1):
            raise ValueError('Decoder input sequence length should be 1')

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs) #[64, 1, 14]
        # encoder_outputs [14, 64, 512]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) #[64, 1, 512]

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) #[64, 512]
        context = context.squeeze(1) #[64, 512]
        concat_input = torch.cat((rnn_output, context), 1) #[64, 1024]
        concat_output = torch.tanh(self.concat(concat_input)) #[64, 512]

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output) #[64, output_size]
        output = F.softmax(output, dim=1)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights
