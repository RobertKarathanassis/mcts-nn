#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <cmath>
#include <unordered_map>
#include <memory>
#include <random>

namespace py = pybind11;

constexpr int NUM_QUEEN_DIRS = 8;
constexpr int NUM_QUEEN_DISTS = 7;
constexpr int NUM_QUEEN_MOVES = NUM_QUEEN_DIRS * NUM_QUEEN_DISTS;     // 56
constexpr int NUM_KNIGHT_MOVES = 8;
constexpr int NUM_UNDERPROMO = 9;                                     // 3 * 3
constexpr int MOVE_TYPE_TOTAL = NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES + NUM_UNDERPROMO;
constexpr int ACTION_SIZE = 8 * 8 * MOVE_TYPE_TOTAL;

constexpr int QUEEN_OFFSET = 0;
constexpr int KNIGHT_OFFSET = QUEEN_OFFSET + NUM_QUEEN_MOVES;         // 56
constexpr int UNDERPROMO_OFFSET = KNIGHT_OFFSET + NUM_KNIGHT_MOVES;   // 64

static const int QUEEN_DIRS[NUM_QUEEN_DIRS][2] = {
    { 1,  0}, { 1,  1}, { 0,  1}, {-1,  1},
    {-1,  0}, {-1, -1}, { 0, -1}, { 1, -1},
};
static const int KNIGHT_DELTAS[NUM_KNIGHT_MOVES][2] = {
    { 2,  1}, { 1,  2}, {-1,  2}, {-2,  1},
    {-2, -1}, {-1, -2}, { 1, -2}, { 2, -1},
};
static const int UNDERPROMO_DELTAS[3] = {-1, 0, 1};
static const int UNDERPROMO_PIECES[3] = {2, 3, 4}; // KNIGHT, BISHOP, ROOK

inline int mirror_if_black(int sq, bool color) {
    return color ? sq : sq ^ 56; // python-chess.square_mirror
}

int encode_move_cpp(const py::object& board, const py::object& move) {
    bool side = board.attr("turn").cast<bool>();
    int from_sq = move.attr("from_square").cast<int>();
    int to_sq   = move.attr("to_square").cast<int>();
    from_sq = mirror_if_black(from_sq, side);
    to_sq   = mirror_if_black(to_sq, side);

    int fr = from_sq / 8, ff = from_sq % 8;
    int tr = to_sq / 8, tf = to_sq % 8;
    int dy = tr - fr, dx = tf - ff;

    py::object promo_obj = move.attr("promotion");
    int promo = promo_obj.is_none() ? 0 : promo_obj.cast<int>();

    if (promo == 0 || promo == 5) {
        int dir_y = (dy > 0) - (dy < 0);
        int dir_x = (dx > 0) - (dx < 0);
        for (int i = 0; i < NUM_QUEEN_DIRS; ++i) {
            if (QUEEN_DIRS[i][0] == dir_y && QUEEN_DIRS[i][1] == dir_x) {
                int dist = std::max(std::abs(dy), std::abs(dx));
                if (dy == dir_y * dist && dx == dir_x * dist && dist >= 1 && dist <= 7) {
                    int mtype = QUEEN_OFFSET + i * 7 + (dist - 1);
                    return fr * 8 * MOVE_TYPE_TOTAL + ff * MOVE_TYPE_TOTAL + mtype;
                }
            }
        }
    }

    if (promo == 0) {
        for (int i = 0; i < NUM_KNIGHT_MOVES; ++i) {
            if (KNIGHT_DELTAS[i][0] == dy && KNIGHT_DELTAS[i][1] == dx) {
                int mtype = KNIGHT_OFFSET + i;
                return fr * 8 * MOVE_TYPE_TOTAL + ff * MOVE_TYPE_TOTAL + mtype;
            }
        }
    }

    if (promo != 0 && promo != 5 && fr == 6 && dy == 1) {
        int delta_i = -1, piece_i = -1;
        for (int i = 0; i < 3; ++i) if (UNDERPROMO_DELTAS[i] == dx) delta_i = i;
        for (int i = 0; i < 3; ++i) if (UNDERPROMO_PIECES[i] == promo) piece_i = i;
        if (delta_i != -1 && piece_i != -1) {
            int mtype = UNDERPROMO_OFFSET + delta_i * 3 + piece_i;
            return fr * 8 * MOVE_TYPE_TOTAL + ff * MOVE_TYPE_TOTAL + mtype;
        }
    }

    return -1;
}

py::object decode_index_cpp(const py::object& board, int index) {
    int fr = index / (8 * MOVE_TYPE_TOTAL);
    int ff = (index / MOVE_TYPE_TOTAL) % 8;
    int mtype = index % MOVE_TYPE_TOTAL;

    int from_sq = ff + fr * 8;
    int to_sq = -1;
    int promo = 0;

    if (mtype < KNIGHT_OFFSET) {
        int rel = mtype - QUEEN_OFFSET;
        int dir_i = rel / 7;
        int dist = rel % 7 + 1;
        int dy = QUEEN_DIRS[dir_i][0];
        int dx = QUEEN_DIRS[dir_i][1];
        to_sq = (ff + dx * dist) + (fr + dy * dist) * 8;
        if (fr == 6 && dy == 1) {
            py::object piece = board.attr("piece_type_at")(mirror_if_black(from_sq, board.attr("turn").cast<bool>()));
            int piece_type = piece.is_none() ? 0 : piece.cast<int>();
            promo = (piece_type == 1) ? 5 : 0; // auto queen if pawn advance
        }
    } else if (mtype < UNDERPROMO_OFFSET) {
        int rel = mtype - KNIGHT_OFFSET;
        int dy = KNIGHT_DELTAS[rel][0];
        int dx = KNIGHT_DELTAS[rel][1];
        to_sq = (ff + dx) + (fr + dy) * 8;
    } else {
        int rel = mtype - UNDERPROMO_OFFSET;
        int dx = UNDERPROMO_DELTAS[rel / 3];
        promo = UNDERPROMO_PIECES[rel % 3];
        to_sq = (ff + dx) + (fr + 1) * 8;
    }

    bool side = board.attr("turn").cast<bool>();
    from_sq = mirror_if_black(from_sq, side);
    to_sq = mirror_if_black(to_sq, side);

    py::object move_cls = py::module::import("chess").attr("Move");
    if (promo == 0)
        return move_cls(from_sq, to_sq);
    return move_cls(from_sq, to_sq, py::arg("promotion") = promo);
}

struct Node {
    Node* parent;
    bool to_play;
    float prior;
    int n;
    float w;
    std::unordered_map<int, std::unique_ptr<Node>> children;

    Node(Node* p, bool t, float pr)
        : parent(p), to_play(t), prior(pr), n(0), w(0.0f) {}

    float q() const { return n == 0 ? 0.f : w / n; }
    float u(float c_puct, int n_parent) const {
        return c_puct * prior * std::sqrt(static_cast<float>(n_parent)) / (1 + n);
    }
};

class MCTS {
public:
    MCTS(py::object net, float c_puct=1.5f,
         float dir_alpha=0.3f, float dir_eps=0.25f, int sims=800)
        : net_(std::move(net)), c_puct_(c_puct),
          dir_a_(dir_alpha), dir_eps_(dir_eps), sims_(sims) {
        board_to_tensor_ = py::module::import("data.board").attr("board_to_tensor");
        action_size_ = ACTION_SIZE;
        auto params = net_.attr("parameters")();
        py::object first = py::iter(params).next();
        device_ = first.attr("device");
    }

    py::dict run(py::object board);

private:
    std::pair<int, Node*> select_child(Node* node);
    float expand(Node* node, py::object board);
    void add_dirichlet_noise(Node& root);

    py::object net_;
    py::object board_to_tensor_;
    py::object device_;
    int action_size_;
    std::mt19937 rng_{std::random_device{}()};
    float c_puct_;
    float dir_a_;
    float dir_eps_;
    int sims_;
};

std::pair<int, Node*> MCTS::select_child(Node* node) {
    int n_parent = node->n;
    int best_idx = -1;
    Node* best_child = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();
    for (auto& kv : node->children) {
        int idx = kv.first;
        Node* child = kv.second.get();
        float score = child->q() + child->u(c_puct_, n_parent);
        if (score > best_score) {
            best_idx = idx;
            best_child = child;
            best_score = score;
        }
    }
    return {best_idx, best_child};
}

float MCTS::expand(Node* node, py::object board) {
    if (py::bool_(board.attr("is_game_over")())) {
        std::string result = board.attr("result")().cast<std::string>();
        if (result == "1-0") {
            return node->to_play ? 1.f : -1.f;
        } else if (result == "0-1") {
            return node->to_play ? -1.f : 1.f;
        } else {
            return 0.f;
        }
    }

    py::object planes = board_to_tensor_(board);
    py::object tensor = planes;
    if (py::hasattr(planes, "unsqueeze"))
        tensor = planes.attr("unsqueeze")(0);

    // Move tensor to same device as network
    tensor = tensor.attr("to")(device_, py::arg("non_blocking")=true);

    py::object out = net_(tensor);
    py::tuple tup = out.cast<py::tuple>();
    torch::Tensor logits = tup[0].cast<torch::Tensor>();
    torch::Tensor value_t = tup[1].cast<torch::Tensor>();
    logits = logits.squeeze().cpu();
    auto logits_arr = logits.to(torch::kCPU).contiguous();
    std::vector<float> logits_vec(logits_arr.data_ptr<float>(), logits_arr.data_ptr<float>() + logits_arr.numel());
    float value = value_t.item<float>();

    std::vector<float> priors(action_size_, 0.f);
    py::object legal_moves = board.attr("legal_moves");
    for (auto mv : legal_moves) {
        int idx = encode_move_cpp(board, mv);
        if (idx >= 0)
            priors[idx] = logits_vec[idx];
    }
    // softmax
    float maxlog = *std::max_element(priors.begin(), priors.end());
    float sum = 0.f;
    for (float& l : priors) {
        l = std::exp(l - maxlog);
        sum += l;
    }
    for (float& l : priors) l /= sum;

    // populate children
    for (auto mv : legal_moves) {
        int idx = encode_move_cpp(board, mv);
        if (idx >= 0)
            node->children[idx] = std::make_unique<Node>(node, !bool(board.attr("turn").cast<bool>()), priors[idx]);
    }
    return value;
}

void MCTS::add_dirichlet_noise(Node& root) {
    if (root.children.empty()) return;
    std::gamma_distribution<float> gamma(dir_a_, 1.f);
    std::vector<float> noise;
    noise.reserve(root.children.size());
    float sum = 0.f;
    for (size_t i = 0; i < root.children.size(); ++i) {
        float x = gamma(rng_);
        noise.push_back(x);
        sum += x;
    }
    for (float& n : noise) n /= sum;
    size_t i = 0;
    for (auto& kv : root.children) {
        kv.second->prior = (1 - dir_eps_) * kv.second->prior + dir_eps_ * noise[i++];
    }
}

py::dict MCTS::run(py::object board) {
    py::gil_scoped_release release;
    Node root(nullptr, board.attr("turn").cast<bool>(), 0.f);
    expand(&root, board);
    add_dirichlet_noise(root);

    bool root_turn = board.attr("turn").cast<bool>();

    for (int s=0; s<sims_; ++s) {
        py::object scratch = board.attr("copy")(py::arg("stack")=false);
        py::object push_fn = scratch.attr("push");
        std::vector<Node*> path{&root};
        Node* node = &root;
        while (!node->children.empty()) {
            auto sel = select_child(node);
            int idx = sel.first;
            node = sel.second;
            push_fn(decode_index_cpp(scratch, idx));
            path.push_back(node);
        }
        float value = expand(node, scratch);
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            Node* n = *it;
            n->n += 1;
            n->w += (n->to_play == root_turn) ? value : -value;
        }
    }
    py::dict visits;
    for (auto& kv : root.children) {
        visits[py::int_(kv.first)] = py::int_(kv.second->n);
    }
    return visits;
}

PYBIND11_MODULE(mcts_cpp, m) {
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<py::object, float, float, float, int>(),
             py::arg("net"), py::arg("c_puct")=1.5f,
             py::arg("dir_alpha")=0.3f, py::arg("dir_eps")=0.25f,
             py::arg("sims")=800)
        .def("run", &MCTS::run);
}

