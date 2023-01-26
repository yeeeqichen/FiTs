import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
from utils.utils import freeze_net
from torch import _softmax_backward_data
from transformers import RobertaTokenizer


class AlignMutualInfo(nn.Module):
    def __init__(self, lm_embed_size, gnn_embed_size, hidden_size):
        super(AlignMutualInfo, self).__init__()
        self.lm_transform = torch.nn.Sequential(
            torch.nn.Linear(lm_embed_size, hidden_size)
        )
        self.gnn_transform = torch.nn.Sequential(
            torch.nn.Linear(gnn_embed_size, hidden_size)
        )
        self.N = 1.
        self.M = 11008.

    def _cal(self, lm_inputs, gnn_inputs, neg=False):
        lm_inputs = self.lm_transform(lm_inputs)
        gnn_inputs = self.gnn_transform(gnn_inputs)

        lm_inputs = lm_inputs / torch.norm(lm_inputs, p=2, dim=-1).unsqueeze(1)
        gnn_inputs = gnn_inputs / torch.norm(gnn_inputs, p=2, dim=-1).unsqueeze(1)

        logits = torch.exp(torch.diagonal(torch.matmul(lm_inputs, gnn_inputs.transpose(1, 0))))

        if not neg:
            return torch.log(logits / (logits + self.N / self.M))
        return torch.log(1 - (logits / (logits + self.N / self.M)))

    def forward(self, lm_embeds, gnn_embeds, neg_gnn_embeds):
        pos_scores = self._cal(lm_embeds, gnn_embeds)
        neg_scores = self._cal(lm_embeds, neg_gnn_embeds, True)

        pos = torch.mean(pos_scores)
        neg = torch.mean(neg_scores)

        loss = -pos - neg
        return loss


def test_mutual():
    lm_in = torch.rand(10, 1024)
    gnn_in = torch.rand(10, 200)
    neg_in = torch.rand(10, 200)
    layer = AlignMutualInfo(1024, 200, 512)
    res = layer(lm_in, gnn_in, neg_in)
    print(res.shape)
    print(res)


class AlignSemantic(nn.Module):
    def __init__(self, lm_embed_size, gnn_embed_size, proj_size=512, loss_type='mse'):
        super(AlignSemantic, self).__init__()
        self.lm_transform = torch.nn.Sequential(
            torch.nn.Linear(lm_embed_size, lm_embed_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(lm_embed_size, eps=1e-12),
            torch.nn.Linear(lm_embed_size, proj_size)
        )
        self.gnn_transform = torch.nn.Sequential(
            torch.nn.Linear(gnn_embed_size, gnn_embed_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(gnn_embed_size, eps=1e-12),
            torch.nn.Linear(gnn_embed_size, proj_size)
        )
        self.loss_type = loss_type
        if loss_type == 'mse':
            self.loss_fn = torch.nn.MSELoss()
            self.margin = 0.2
        elif loss_type == 'cosine':
            self.margin = 0.2
            self.loss_fn = torch.nn.CosineSimilarity()
        else:
            raise KeyError

    def forward(self, lm_inputs, gnn_inputs, neg_inputs):
        lm_semantic = self.lm_transform(lm_inputs)
        gnn_semantic = self.gnn_transform(gnn_inputs)
        neg_semantic = self.gnn_transform(neg_inputs)

        pos_score = self.loss_fn(lm_semantic, gnn_semantic)
        neg_score = self.loss_fn(lm_semantic, neg_semantic)

        if self.loss_type == 'cosine':
            pos_score = torch.sum(1 - pos_score)
            neg_score = torch.sum(1 - neg_score)
            return pos_score - neg_score
            # return max(0, self.margin - (neg_score - pos_score))
        elif self.loss_type == 'mse':
            loss = self.margin - (neg_score - pos_score)
            if loss < 0:
                loss += -loss
            return loss
        else:
            raise KeyError


def test_align_semantic():
    lm_in = torch.rand(100, 1024)
    gnn_in = torch.rand(100, 200)
    neg_in = torch.rand(100, 200)
    layer = AlignSemantic(1024, 200, loss_type='cosine')
    res = layer(lm_in, gnn_in, neg_in)
    print(res.shape)
    print(res)


class AlignClassifier(nn.Module):
    def __init__(self, embed_size):
        super(AlignClassifier, self).__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(embed_size, embed_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(embed_size, eps=1e-12)
        )
        self.cls_head = torch.nn.Linear(embed_size, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        logits = self.cls_head(self.transform(inputs))
        prob = torch.softmax(logits, dim=-1)
        return self.loss_fn(prob, labels)


class AlignMaskPool(nn.Module):
    def __init__(self, mean=False):
        super(AlignMaskPool, self).__init__()
        self.mean = mean

    def forward(self, inputs):
        if self.mean:
            return torch.mean(inputs, dim=0)
        return torch.max(inputs, dim=0).values


class MaskedTokenCls(nn.Module):
    def __init__(self, embed_size):
        super(MaskedTokenCls, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.vocab_size = tokenizer.vocab_size
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(embed_size, embed_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(embed_size, eps=1e-12)
        )
        self.lm_head = torch.nn.Linear(embed_size, self.vocab_size)

    def forward(self, masked_token_embed, origin_token_ids):
        pred = self.lm_head(self.transform(masked_token_embed)).view(-1, self.vocab_size)
        # print(pred.shape)
        pred = torch.softmax(pred, dim=-1)
        # print(pred.shape)
        return self.loss_fn(pred, origin_token_ids.view(-1))
        # raise NotImplementedError


def test_mask_token():
    input = torch.rand(128, 100, 256)
    token_ids = torch.randint(high=1000, size=(128, 100))
    layer = MaskedTokenCls(256)
    res = layer(input, token_ids)
    print(res.shape)
    print(res)


class MaskedNodeSimilarity(nn.Module):
    def __init__(self):
        super(MaskedNodeSimilarity, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, ori_embed, masked_embed):
        sim = self.cos(ori_embed, masked_embed).view(-1)
        return torch.sum(1 - sim)


def test_masked_node_sim():
    ori_embed = torch.rand(200, 128)
    masked_embed = torch.rand(200, 128)
    layer = MaskedNodeSimilarity()
    res = layer(ori_embed, masked_embed)
    print(res)


class NodeRegularization(nn.Module):
    def __init__(self, thresh_hold=10, max_cnt=20, random_choice=False, max_node_num=None):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.thresh_hold = thresh_hold
        self.max_cnt = max_cnt
        self.random_choice = random_choice
        self.cache = {
            'train': dict(),
            'dev': dict(),
            'test': dict(),
        }
        assert max_node_num is not None
        self.max_node_num = max_node_num

    def forward(self, node_embed, edge_embed, node_scores, edge_idx, labels=None, mini_batch_id=None, purpose=None):
        assert purpose is not None and purpose in self.cache
        assert mini_batch_id is not None
        if purpose != 'test':
            assert labels is not None
        if self.random_choice:
            n_edges = edge_idx.shape[1]
            target_cnt = min(self.max_cnt, n_edges)
            indices = torch.tensor(np.random.randint(low=0, high=n_edges, size=target_cnt)).to(node_embed.device)
            # indices = torch.tensor(random.sample(range(n_edges), target_cnt)).to(node_embed.device)
            selected_edge_embed = edge_embed[indices]
            selected_source_embed = node_embed[edge_idx[0][indices]]
            selected_target_embed = node_embed[edge_idx[1][indices]]
        else:
            selected_edges = []
            if mini_batch_id in self.cache[purpose]:
                selected_edges = self.cache[purpose][mini_batch_id]
            else:
                # target_node_id = set(
                #     [int(_.detach().cpu()) for _ in torch.argsort(node_scores.view(-1))[-self.thresh_hold:]])
                n_nodes = len(node_scores.view(-1))
                # print(n_nodes, self.max_node_num)
                # print(labels.shape)
                target_node_id = set(torch.tensor(
                    list(range(0, n_nodes))).view(-1, self.max_node_num)[labels == 1].view(-1).tolist())
                # print(target_node_id)
                cnt = 0
                for i, (source_idx, target_idx) in enumerate(zip(edge_idx[0], edge_idx[1])):
                    if int(source_idx.detach().cpu()) in target_node_id or int(
                            target_idx.detach().cpu()) in target_node_id:
                        cnt += 1
                        selected_edges.append(i)
                        if cnt == self.max_cnt:
                            break
                self.cache[purpose][mini_batch_id] = selected_edges
            if len(selected_edges) == 0:
                return torch.tensor(0.0).to(node_embed.device)
            # print(len(selected_edges))
            selected_edge_embed = edge_embed[selected_edges]
            selected_source_embed = node_embed[edge_idx[0][selected_edges]]
            selected_target_embed = node_embed[edge_idx[1][selected_edges]]

        if selected_source_embed.shape[0] == 0:
            return 0
        sim = self.cos(selected_source_embed + selected_edge_embed, selected_target_embed).view(-1)
        return torch.sum(1 - sim)


def test_node_regularization():
    layer = NodeRegularization(random_choice=False, max_node_num=20)
    source_node_embed = torch.rand(200, 200)
    edge_embed = torch.rand(3, 200)
    node_scores = torch.rand(200)
    edge_idx = torch.tensor([[0, 1, 2], [2, 1, 0]])
    label = torch.tensor([0, 2])
    label = torch.nn.functional.one_hot(label, num_classes=5).view(-1)
    print(label)
    # exit()
    # label = torch.tensor([1, 0, 0, 0, 1])
    print(layer(source_node_embed, edge_embed, node_scores, edge_idx, labels=label,
                purpose='train', mini_batch_id=1))


class ScoreRegression(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_class=1, num_layers=1, dropout_rate=0.2):
        super().__init__()
        self.mlp = MLP(
            input_size=node_dim,
            hidden_size=hidden_dim,
            output_size=num_class,
            num_layers=num_layers,
            dropout=dropout_rate
        )

    def forward(self, node_feats):
        '''
        :param node_feats: (batch_size, max_node_num, node_dim)
        :return:
        '''
        return self.mlp(node_feats)


def test_score_regression():
    classifier = ScoreRegreession(
        node_dim=200,
        hidden_dim=256,
    )
    loss_fn = torch.nn.MSELoss()
    node_feats = torch.rand(2, 2, 200, dtype=torch.float32)
    labels = torch.tensor([[1, 2], [3, 4]])
    pred = classifier(node_feats)
    print(pred)
    print(loss_fn(pred.squeeze(-1), labels))


class NodeClassifier(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_class, num_layers=1, dropout_rate=0.2):
        super().__init__()
        self.mlp = MLP(
            input_size=node_dim,
            hidden_size=hidden_dim,
            output_size=num_class,
            num_layers=num_layers,
            dropout=dropout_rate
        )

    def forward(self, node_feats):
        '''
        :param node_feats: (batch_size, max_node_num, node_dim)
        :return:
        '''
        return F.softmax(self.mlp(node_feats), dim=-1)


def test_node_classifier():
    classifier = NodeClassifier(
        node_dim=200,
        hidden_dim=256,
        num_class=4
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    node_feats = torch.rand(2, 2, 200, dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 3])
    pred = classifier(node_feats)
    print(pred)
    print(loss_fn(pred.view(-1, 4), labels))


class CrossAttention(nn.Module):
    def __init__(self, num_attention_heads=4, seq1_hidden_size=4, seq2_hidden_size=4,
                 position_embedding_type=None, attention_probs_dropout_prob=0.2):
        super().__init__()
        hidden_size = seq2_hidden_size
        assert hidden_size % num_attention_heads == 0

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(seq1_hidden_size, self.all_head_size)
        self.key = nn.Linear(seq2_hidden_size, self.all_head_size)
        self.value = nn.Linear(seq2_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            exit('error position embedding')
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        sequence1_hidden_states,
        sequence2_hidden_states,
        output_attentions=False,
    ):
        key_layer = self.transpose_for_scores(self.key(sequence2_hidden_states))
        value_layer = self.transpose_for_scores(self.value(sequence2_hidden_states))
        query_layer = self.transpose_for_scores(self.query(sequence1_hidden_states))


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        #     seq_length = hidden_states.size()[1]
        #     position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        #     position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        #     distance = position_ids_l - position_ids_r
        #     positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        #     positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility
        #
        #     if self.position_embedding_type == "relative_key":
        #         relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        #         attention_scores = attention_scores + relative_position_scores
        #     elif self.position_embedding_type == "relative_key_query":
        #         relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        #         relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
        #         attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)


        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


def test_cross_attn():
    seq1_hidden_states = torch.rand(32, 128, 1024)
    seq2_hidden_states = torch.rand(32, 128, 200)
    cross_attn_layer1 = CrossAttention(
        seq1_hidden_size=1024,
        seq2_hidden_size=200
    )
    cross_attn_layer2 = CrossAttention(
        seq1_hidden_size=200,
        seq2_hidden_size=1024
    )
    output1 = cross_attn_layer1(seq1_hidden_states, seq2_hidden_states)[0]
    output2 = cross_attn_layer2(seq2_hidden_states, seq1_hidden_states)[0]
    print(output1.shape, output2.shape)


class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory
    Args:
        input (`torch.tensor`): The input tensor that will apply softmax.
        mask (`torch.IntTensor`):
            The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax
    Example:
    ```python
    >>> import torch
    >>> from transformers.models.deberta.modeling_deberta import XSoftmax
    >>> # Make a tensor
    >>> x = torch.randn([4, 20, 100])
    >>> # Create a mask
    >>> mask = (x > 0).int()
    >>> # Specify the dimension to apply softmax
    >>> dim = -1
    >>> y = XSoftmax.apply(x, mask, dim)
    ```"""

    @staticmethod
    def forward(self, input, dim):
        # print(input.shape, mask.shape)
        self.dim = dim
        # rmask = ~(mask.bool())
        #
        # output = input.masked_fill(rmask, float("-inf"))
        output = torch.softmax(input, self.dim)
        # output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        (output,) = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
        return inputGrad, None, None

    @staticmethod
    def symbolic(g, self, mask, dim):
        import torch.onnx.symbolic_helper as sym_help
        from torch.onnx.symbolic_opset9 import masked_fill, softmax

        mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        r_mask = g.op(
            "Cast",
            g.op("Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value),
            to_i=sym_help.cast_pytorch_to_onnx["Byte"],
        )
        output = masked_fill(g, self, r_mask, g.op("Constant", value_t=torch.tensor(float("-inf"))))
        output = softmax(g, output, dim)
        return masked_fill(g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.uint8)))


class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training
    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module
        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


class DisentangledSelfAttention(nn.Module):
    """
    Disentangled self-attention module
    Parameters:
        config (`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaConfig`]
    """

    def __init__(self, context_dim=1, pos_dim=1, hidden_dim=1, num_head=4, drop_out_prob=0.1,
                 relative_attn=True, pos_att_type='c2pp2c'):
        super().__init__()
        # if config.hidden_size % config.num_attention_heads != 0:
        #     raise ValueError(
        #         f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
        #         f"heads ({config.num_attention_heads})"
        #     )
        assert hidden_dim % num_head == 0
        self.num_attention_heads = num_head
        self.attention_head_size = int(hidden_dim / num_head)
        self.all_head_size = hidden_dim
        self.V_proj = nn.Linear(context_dim, context_dim, bias=True)
        self.context_QK_proj = nn.Linear(context_dim, hidden_dim * 2, bias=False)
        self.q_bias = nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        # self.v_bias = nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        self.pos_att_type = pos_att_type if pos_att_type is not None else []

        self.relative_attention = relative_attn

        if self.relative_attention:
            # self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            # if self.max_relative_positions < 1:
            #     self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = StableDropout(0.1)

            if "c2p" in self.pos_att_type:
                self.pos_proj = nn.Linear(pos_dim, self.all_head_size, bias=False)
            if "p2c" in self.pos_att_type:
                self.pos_q_proj = nn.Linear(pos_dim, self.all_head_size)

        self.dropout = StableDropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        rel_embeddings=None,
    ):
        qk = self.context_QK_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
        v = self.V_proj(hidden_states)
        query_layer, key_layer = self.transpose_for_scores(qk).chunk(2, dim=-1)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1 + len(self.pos_att_type)
        scale = math.sqrt(query_layer.size(-1) * scale_factor)
        query_layer = query_layer / scale
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, rel_embeddings, scale_factor)
            # print(rel_att.shape)
            # exit()
        # print(attention_scores.shape, rel_att.shape)
        if rel_att is not None:
            attention_scores = attention_scores + rel_att

        # bxhxlxd

        attention_probs = XSoftmax.apply(attention_scores, -1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

    def disentangled_att_bias(self, query_layer, key_layer, rel_embeddings, scale_factor):
        score = 0

        # content->position
        if "c2p" in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            # c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            # c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            score += c2p_att

        # position->content
        if "p2c" in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1) * scale_factor)
            # if query_layer.size(-2) != key_layer.size(-2):
            #     r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            # else:
            #     r_pos = relative_pos
            # p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            # p2c_att = torch.gather(
            #     p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)
            # ).transpose(-1, -2)
            #
            # if query_layer.size(-2) != key_layer.size(-2):
            #     pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
            #     p2c_att = torch.gather(p2c_att, dim=-2, index=pos_dynamic_expand(pos_index, p2c_att, key_layer))
            score += p2c_att

        return score


def test_deberta():
    from transformers import DebertaConfig
    config = DebertaConfig()
    config.relative_attention = True
    config.pos_att_type = 'p2c'
    print(config)
    # exit()
    layer1 = DisentangledSelfAttention(
        context_dim=1024,
        pos_dim=200,
        hidden_dim=256,
        num_head=4
    )
    layer2 = DisentangledSelfAttention(
        context_dim=200,
        pos_dim=1024,
        hidden_dim=256,
        num_head=4
    )
    hidden_states = torch.rand((24, 32, 1024))
    rel_embeddings = torch.rand((24, 32, 200))
    print(layer1, layer2)
    res1 = layer1(hidden_states, rel_embeddings)
    res2 = layer2(rel_embeddings, hidden_states)
    print(res1.shape, res2.shape)


class DeBERTaIE(nn.Module):
    def __init__(self, context_dim, pos_dim, hidden_dim, num_heads, drop_out_prob=0.1, pos_att_type='p2c+c2p',
                 relative_attn=True):
        super().__init__()
        self.ie = DisentangledSelfAttention(context_dim,
                                            pos_dim,
                                            hidden_dim,
                                            num_heads,
                                            drop_out_prob=drop_out_prob,
                                            pos_att_type=pos_att_type,
                                            relative_attn=relative_attn)

    def forward(self, context_feats, pos_feats):
        return self.ie(context_feats, pos_feats)


class CNNContextNodePooler(nn.Module):
    def __init__(self, feat_dim=1):
        super().__init__()
        self.cnn = nn.Sequential(
            # nn.Conv1d(
            #     in_channels=feat_dim,
            #     out_channels=feat_dim,
            #     kernel_size=5,
            #     stride=2
            # ),
            # nn.MaxPool1d(
            #     kernel_size=2
            # ),
            nn.Conv1d(
              in_channels=feat_dim,
              out_channels=feat_dim,
              kernel_size=3,
              stride=2
            )
        )

    def forward(self, all_context_node_feats):
        """
        :param all_context_node_feats: (layer_num, batch_size, feat_dim)
        :return: (batch_size, feat_dim)
        """
        all_context_node_feats = all_context_node_feats.permute(1, 2, 0)
        # print(all_context_node_feats.shape)
        conv_res = self.cnn(all_context_node_feats)
        conv_res = conv_res.max(dim=-1)[0]
        return conv_res


class IterativeIELayer(nn.Module):
    def __init__(self, q_dim, k_dim, ie_dim, num_heads=2, p_fc=0.2, ie_layer_num=1):
        super().__init__()
        self.attn_pool = AttPoolLayer(
            d_q=q_dim,
            d_k=k_dim,
            # n_head=num_heads
        )
        self.ffn = MLP(
            input_size=q_dim + k_dim,
            hidden_size=ie_dim,
            output_size=q_dim,
            dropout=p_fc,
            num_layers=ie_layer_num
        )

    def forward(self, sequence_feats, node_or_token_feat):
        """
        :param sequence_feats: the lm_feats or gnn_feats,
               shape: (batch_size, max_seq_length or max_nodes, lm_dim or gnn_dim)
        :param node_or_token_feat: shape: (batch_size, lm_dim or gnn_dim
        :return: new node or token feat, shape: (batch_size, lm_dim or gnn_dim)
        """
        pooled_seq_feat = self.attn_pool(
            q=node_or_token_feat,
            k=sequence_feats
        )[0]
        concat_feats = torch.cat([pooled_seq_feat, node_or_token_feat], dim=-1)
        new_node_or_token_feat = self.ffn(concat_feats)
        return new_node_or_token_feat


def test_iterative_ie_layer():
    seq_feats = torch.rand(32, 128, 200)
    node_or_token_feat = torch.rand(32, 1024)
    ie_layer = IterativeIELayer(
        q_dim=1024,
        k_dim=200,
        ie_dim=400,
        num_heads=2
    )
    output = ie_layer(seq_feats, node_or_token_feat)
    print(output.shape)



def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class TypedLinear(nn.Linear):
    def __init__(self, in_features, out_features, n_type):
        super().__init__(in_features, n_type * out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.n_type = n_type

    def forward(self, X, type_ids=None):
        """
        X: tensor of shape (*, in_features)
        type_ids: long tensor of shape (*)
        """
        output = super().forward(X)
        if type_ids is None:
            return output
        output_shape = output.size()[:-1] + (self.out_features,)
        output = output.view(-1, self.n_type, self.out_features)
        idx = torch.arange(output.size(0), dtype=torch.long, device=type_ids.device)
        output = output[idx, type_ids.view(-1)].view(*output_shape)
        return output


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)


class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
        else:
            mask = mask_or_lengths
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = masked_inputs.max(1)[0]
        return max_pooled


class MeanPoolLayer(nn.Module):
    """
    A layer that performs mean pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
            lengths = mask_or_lengths.float()
        else:
            mask, lengths = mask_or_lengths, (1 - mask_or_lengths.float()).sum(1)
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), 0.0)
        mean_pooled = masked_inputs.sum(1) / lengths.unsqueeze(-1)
        return mean_pooled


def dropout_mask(x, sz, p: float):
    """
    Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)


class EmbeddingDropout(nn.Module):
    """
    Apply dropout with probabily `embed_p` to an embedding layer `emb`.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """

    def __init__(self, emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)


class RNNDropout(nn.Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m


class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size=300, emb_size=300, hidden_size=300, num_layers=2, bidirectional=True,
                 emb_p=0, input_p=0, hidden_p=0, output_p=0, pretrained_emb=None, pooling=True, pad=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.emb_p = emb_p
        self.input_p = input_p
        self.hidden_p = hidden_p
        self.output_p = output_p
        self.pooling = pooling

        self.emb = EmbeddingDropout(nn.Embedding(vocab_size, emb_size), emb_p)
        if pretrained_emb is not None:
            self.emb.emb.weight.data.copy_(pretrained_emb)
        else:
            bias = np.sqrt(6.0 / emb_size)
            nn.init.uniform_(self.emb.emb.weight, -bias, bias)
        self.input_dropout = nn.Dropout(input_p)
        self.output_dropout = nn.Dropout(output_p)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=(hidden_size // 2 if self.bidirectional else hidden_size),
                           num_layers=num_layers, dropout=hidden_p, bidirectional=bidirectional,
                           batch_first=True)
        self.max_pool = MaxPoolLayer()

    def forward(self, inputs, lengths):
        """
        inputs: tensor of shape (batch_size, seq_len)
        lengths: tensor of shape (batch_size)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bz, full_length = inputs.size()
        embed = self.emb(inputs)
        embed = self.input_dropout(embed)
        lstm_inputs = pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
        rnn_outputs, _ = self.rnn(lstm_inputs)
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=True, total_length=full_length)
        rnn_outputs = self.output_dropout(rnn_outputs)
        return self.max_pool(rnn_outputs, lengths) if self.pooling else rnn_outputs


class TripleEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, input_p, output_p, hidden_p, num_layers, bidirectional=True, pad=False,
                 concept_emb=None, relation_emb=None
                 ):
        super().__init__()
        if pad:
            raise NotImplementedError
        self.input_p = input_p
        self.output_p = output_p
        self.hidden_p = hidden_p
        self.cpt_emb = concept_emb
        self.rel_emb = relation_emb
        self.input_dropout = nn.Dropout(input_p)
        self.output_dropout = nn.Dropout(output_p)
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=(hidden_dim // 2 if self.bidirectional else hidden_dim),
                          num_layers=num_layers, dropout=hidden_p, bidirectional=bidirectional,
                          batch_first=True)

    def forward(self, inputs):
        '''
        inputs: (batch_size, seq_len)

        returns: (batch_size, h_dim(*2))
        '''
        bz, sl = inputs.size()
        h, r, t = torch.chunk(inputs, 3, dim=1)  # (bz, 1)

        h, t = self.input_dropout(self.cpt_emb(h)), self.input_dropout(self.cpt_emb(t))  # (bz, 1, dim)
        r = self.input_dropout(self.rel_emb(r))
        inputs = torch.cat((h, r, t), dim=1)  # (bz, 3, dim)
        rnn_outputs, _ = self.rnn(inputs)  # (bz, 3, dim)
        if self.bidirectional:
            outputs_f, outputs_b = torch.chunk(rnn_outputs, 2, dim=2)
            outputs = torch.cat((outputs_f[:, -1, :], outputs_b[:, 0, :]), 1)  # (bz, 2 * h_dim)
        else:
            outputs = rnn_outputs[:, -1, :]

        return self.output_dropout(outputs)


class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn


class AttPoolLayer(nn.Module):

    def __init__(self, d_q, d_k, dropout=0.1):
        super().__init__()
        self.w_qs = nn.Linear(d_q, d_k)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q + d_k)))
        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q)
        k: tensor of shape (b, l, d_k)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, d_k)
        """
        qs = self.w_qs(q)  # (b, d_k)
        output, attn = self.attention(qs, k, k, mask=mask)
        output = self.dropout(output)
        return output, attn


class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class TypedMultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1, n_type=1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = TypedLinear(d_k_original, n_head * self.d_k, n_type)
        self.w_vs = TypedLinear(d_k_original, n_head * self.d_v, n_type)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None, type_ids=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: bool tensor of shape (b, l) (optional, default None)
        type_ids: long tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k, type_ids=type_ids).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k, type_ids=type_ids).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class BilinearAttentionLayer(nn.Module):

    def __init__(self, query_dim, value_dim):
        super().__init__()
        self.linear = nn.Linear(value_dim, query_dim, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, query, value, node_mask=None):
        """
        query: tensor of shape (batch_size, query_dim)
        value: tensor of shape (batch_size, seq_len, value_dim)
        node_mask: tensor of shape (batch_size, seq_len)

        returns: tensor of shape (batch_size, value_dim)
        """
        attn = self.linear(value).bmm(query.unsqueeze(-1))
        attn = self.softmax(attn.squeeze(-1))
        if node_mask is not None:
            attn = attn * node_mask
            attn = attn / attn.sum(1, keepdim=True)
        pooled = attn.unsqueeze(1).bmm(value).squeeze(1)
        return pooled, attn


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = True,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # # To limit numerical errors from large vector elements outside the mask, we zero these out.
            # result = nn.functional.softmax(vector * mask, dim=dim)
            # result = result * mask
            # result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            raise NotImplementedError
        else:
            masked_vector = vector.masked_fill(mask.to(dtype=torch.uint8), mask_fill_value)
            result = nn.functional.softmax(masked_vector, dim=dim)
            result = result * (1 - mask)
    return result


class DiffTopK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, k):
        """
        x: tensor of shape (batch_size, n_node)
        k: int
        returns: tensor of shape (batch_size, n_node)
        """
        bs, _ = x.size()
        _, topk_indexes = x.topk(k, 1)  # (batch_size, k)
        output = x.new_zeros(x.size())
        ri = torch.arange(bs).unsqueeze(1).expand(bs, k).contiguous().view(-1)
        output[ri, topk_indexes.view(-1)] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


class SimilarityFunction(nn.Module):
    """
    A ``SimilarityFunction`` takes a pair of tensors with the same shape, and computes a similarity
    function on the vectors in the last dimension.  For example, the tensors might both have shape
    `(batch_size, sentence_length, embedding_dim)`, and we will compute some function of the two
    vectors of length `embedding_dim` for each position `(batch_size, sentence_length)`, returning a
    tensor of shape `(batch_size, sentence_length)`.
    The similarity function could be as simple as a dot product, or it could be a more complex,
    parameterized function.
    """
    default_implementation = 'dot_product'

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        """
        Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
        embedding_dim)``.  Computes a (possibly parameterized) similarity on the final dimension
        and returns a tensor with one less dimension, such as ``(batch_size, length_1, length_2)``.
        """
        raise NotImplementedError


class DotProductSimilarity(SimilarityFunction):
    """
    This similarity function simply computes the dot product between each pair of vectors, with an
    optional scaling to reduce the variance of the output elements.
    Parameters
    ----------
    scale_output : ``bool``, optional
        If ``True``, we will scale the output by ``math.sqrt(tensor.size(-1))``, to reduce the
        variance in the result.
    """

    def __init__(self, scale_output: bool = False) -> None:
        super(DotProductSimilarity, self).__init__()
        self._scale_output = scale_output

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self._scale_output:
            result *= math.sqrt(tensor_1.size(-1))
        return result


class MatrixAttention(nn.Module):
    def __init__(self, similarity_function: SimilarityFunction = None) -> None:
        super().__init__()
        self._similarity_function = similarity_function or DotProductSimilarity()

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        tiled_matrix_1 = matrix_1.unsqueeze(2).expand(matrix_1.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_1.size()[2])
        tiled_matrix_2 = matrix_2.unsqueeze(1).expand(matrix_2.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_2.size()[2])

        return self._similarity_function(tiled_matrix_1, tiled_matrix_2)


class CustomizedEmbedding(nn.Module):
    def __init__(self, concept_num, concept_in_dim, concept_out_dim, use_contextualized=False,
                 pretrained_concept_emb=None, freeze_ent_emb=True, scale=1.0, init_range=0.02,
                 masked_entity_modeling=False):
        super().__init__()
        self.scale = scale
        self.use_contextualized = use_contextualized
        if not use_contextualized:
            if not masked_entity_modeling:
                self.emb = nn.Embedding(concept_num + 2, concept_in_dim)
            else:
                self.emb = nn.Embedding(concept_num + 3, concept_in_dim)
            if pretrained_concept_emb is not None:
                self.emb.weight.data.fill_(0)
                self.emb.weight.data[:concept_num].copy_(pretrained_concept_emb)
            else:
                self.emb.weight.data.normal_(mean=0.0, std=init_range)
            if freeze_ent_emb:
                freeze_net(self.emb)

        if concept_in_dim != concept_out_dim:
            self.cpt_transform = nn.Linear(concept_in_dim, concept_out_dim)
            self.activation = GELU()

    def forward(self, index, contextualized_emb=None):
        """
        index: size (bz, a)
        contextualized_emb: size (bz, b, emb_size) (optional)
        """
        if contextualized_emb is not None:
            assert index.size(0) == contextualized_emb.size(0)
            if hasattr(self, 'cpt_transform'):
                contextualized_emb = self.activation(self.cpt_transform(contextualized_emb * self.scale))
            else:
                contextualized_emb = contextualized_emb * self.scale
            emb_dim = contextualized_emb.size(-1)
            return contextualized_emb.gather(1, index.unsqueeze(-1).expand(-1, -1, emb_dim))
        else:
            if hasattr(self, 'cpt_transform'):
                return self.activation(self.cpt_transform(self.emb(index) * self.scale))
            else:
                return self.emb(index) * self.scale


def run_test():
    print('testing BilinearAttentionLayer...')
    att = BilinearAttentionLayer(100, 20)
    mask = (torch.randn(70, 30) > 0).float()
    mask.requires_grad_()
    v = torch.randn(70, 30, 20)
    q = torch.randn(70, 100)
    o, _ = att(q, v, mask)
    o.sum().backward()
    print(mask.grad)

    print('testing DiffTopK...')
    x = torch.randn(5, 3)
    x.requires_grad_()
    k = 2
    r = DiffTopK.apply(x, k)
    loss = (r ** 2).sum()
    loss.backward()
    assert (x.grad == r * 2).all()
    print('pass')

    a = TripleEncoder()

    triple_input = torch.tensor([[1, 2, 3], [4, 5, 6]])
    res = a(triple_input)
    print(res.size())

    b = LSTMEncoder(pooling=False)
    lstm_inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    lengths = torch.tensor([3, 2])
    res = b(lstm_inputs, lengths)
    print(res.size())


if __name__ == '__main__':
    test_mutual()