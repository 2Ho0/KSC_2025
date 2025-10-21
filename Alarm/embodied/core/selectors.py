import collections
import threading

import numpy as np


# embodied/replay/selectors.py 파일 수정

import collections
import threading
import numpy as np

class RTGBased:
    """
    Trajectory의 시작 RTG(Returns-To-Go) 값에 비례하여 샘플링하는 셀렉터.
    RTG가 높은 '성공' 경험을 더 자주 뽑도록 설계되었습니다.
    """
    def __init__(self, seed=0):
        self.keys = []      # Trajectory의 고유 ID (itemid) 목록
        self.values = []    # 각 trajectory의 시작 RTG 값 목록
        self.indices = {}   # 빠른 삭제를 위한 {itemid: index} 맵
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.keys)

    def __call__(self):
        with self.lock:
            if not self.keys:
                raise ValueError("Buffer is empty.")
            
            # 1. 저장된 모든 RTG 값을 numpy 배열로 변환
            values_np = np.array(self.values, dtype=np.float32)
            
            # RTG 값에 지수 함수를 적용하여 차이를 극대화
            # 큰 값으로 인한 오버플로우를 막기 위해 max 값을 빼줌 (softmax trick)
            shifted_values = values_np - np.max(values_np)
            probs = np.exp(shifted_values)
            
            # 3. 전체 합으로 나누어 확률 분포로 정규화합니다.
            probs /= np.sum(probs)

            if self.rng.random() < 0.01:  # 디버그용 출력, 1% 확률로 출력
                print(f"RTGBased Debug: Max value={np.max(values_np):.4f}, Max prob={np.max(probs):.4f}, Num samples={len(self.keys)}")
            
            # 4. 계산된 확률(p)에 따라 trajectory의 인덱스를 선택합니다.
            index = self.rng.choice(len(self.keys), p=probs)
            return self.keys[index]

    # Replay 클래스에서 trajectory의 itemid와 시작 rtg 값을 받습니다.
    def __setitem__(self, key, value):
        with self.lock:
            self.indices[key] = len(self.keys)
            self.keys.append(key)
            self.values.append(value)

    def __delitem__(self, key):
        with self.lock:
            # O(1) 시간 복잡도로 효율적인 삭제를 위해 마지막 요소와 교체
            index = self.indices.pop(key)
            last_key = self.keys.pop()
            last_value = self.values.pop()
            if index != len(self.keys):
                self.keys[index] = last_key
                self.values[index] = last_value
                self.indices[last_key] = index

class Latest:
    """
    [수정] 최근에 추가된 키를 배치 단위로 미리 확보하여 중복 없이 반환하는 셀렉터.
    """
    def __init__(self, seed=0):
        self.keys = []
        self.lock = threading.Lock()
        # [수정] 현재 처리 중인 배치의 키를 저장할 큐
        self.batch_queue = collections.deque()

    def __len__(self):
        return len(self.keys)

    def __call__(self):
        with self.lock:
            if not self.keys:
                raise ValueError("Buffer is empty.")

            # [핵심 수정] 현재 배치 큐가 비어있으면, 새로 채웁니다.
            # Replay._sample()은 __call__을 배치 크기만큼 반복 호출합니다.
            # 이 로직은 그 반복의 첫 호출 시에만 실행됩니다.
            if not self.batch_queue:
                # Replay._sample()이 몇 개를 요청할지 미리 알 수 없으므로,
                # 일반적으로 사용되는 배치 크기(예: 64)만큼 미리 가져옵니다.
                # 이는 완벽한 해결책은 아니지만, 대부분의 경우 경쟁 상태를 방지합니다.
                # 더 완벽한 해결을 위해서는 Replay 클래스가 배치 크기를 전달해야 합니다.
                # 여기서는 휴리스틱으로 최근 256개를 가져와 큐에 넣습니다.
                num_to_fetch = min(len(self.keys), 256)
                latest_keys = self.keys[-num_to_fetch:]
                self.batch_queue.extend(reversed(latest_keys)) # 최신 순서대로 pop하기 위해 reversed

            if not self.batch_queue:
                 # keys는 있지만 큐를 채우지 못한 경우 (거의 발생 안함)
                 return self.keys[-1]
            
            # 미리 확보해둔 큐에서 하나씩 꺼내서 반환
            return self.batch_queue.popleft()

    def __setitem__(self, key, stepids):
        with self.lock:
            self.keys.append(key)
            # [수정] 키가 추가될 때 큐를 비워서, 다음 샘플링 시
            # 최신 키 목록으로 큐가 다시 채워지도록 합니다.
            self.batch_queue.clear()

    def __delitem__(self, key):
        with self.lock:
            # list.remove는 느리므로, 더 효율적인 방법으로 교체할 수 있다면 좋습니다.
            # 여기서는 기존 로직을 유지합니다.
            if key in self.keys:
                self.keys.remove(key)
            # 삭제 시에도 큐를 비워 일관성을 유지합니다.
            self.batch_queue.clear()

class Fifo:

  def __init__(self):
    self.queue = collections.deque()

  def __call__(self):
    return self.queue[0]

  def __len__(self):
    return len(self.queue)

  def __setitem__(self, key, stepids):
    self.queue.append(key)

  def __delitem__(self, key):
    if self.queue[0] == key:
      self.queue.popleft()
    else:
      # This is very slow but typically not used.
      self.queue.remove(key)


class Uniform:

  def __init__(self, seed=0):
    self.indices = {}
    self.keys = []
    self.rng = np.random.default_rng(seed)
    self.lock = threading.Lock()

  def __len__(self):
    return len(self.keys)

  def __call__(self):
    with self.lock:
      index = self.rng.integers(0, len(self.keys)).item()
      return self.keys[index]

  def __setitem__(self, key, stepids):
    with self.lock:
      self.indices[key] = len(self.keys)
      self.keys.append(key)

  def __delitem__(self, key):
    with self.lock:
      assert 2 <= len(self), len(self)
      index = self.indices.pop(key)
      last = self.keys.pop()
      if index != len(self.keys):
        self.keys[index] = last
        self.indices[last] = index


class Recency:

  def __init__(self, uprobs, seed=0):
    assert uprobs[0] >= uprobs[-1], uprobs
    self.uprobs = uprobs
    self.tree = self._build(uprobs)
    self.rng = np.random.default_rng(seed)
    self.step = 0
    self.steps = {}
    self.items = {}

  def __len__(self):
    return len(self.items)

  def __call__(self):
    for retry in range(10):
      try:
        age = self._sample(self.tree, self.rng)
        if len(self.items) < len(self.uprobs):
          age = int(age / len(self.uprobs) * len(self.items))
        return self.items[self.step - 1 - age]
      except KeyError:
        # Item might have been deleted very recently.
        if retry < 9:
          import time
          time.sleep(0.01)
        else:
          raise

  def __setitem__(self, key, stepids):
    self.steps[key] = self.step
    self.items[self.step] = key
    self.step += 1

  def __delitem__(self, key):
    step = self.steps.pop(key)
    del self.items[step]

  def _sample(self, tree, rng, bfactor=16):
    path = []
    for level, prob in enumerate(tree):
      p = prob
      for segment in path:
        p = p[segment]
      index = rng.choice(len(segment), p=p)
      path.append(index)
    index = sum(
        index * bfactor ** (len(tree) - level - 1)
        for level, index in enumerate(path))
    return index

  def _build(self, uprobs, bfactor=16):
    assert np.isfinite(uprobs).all(), uprobs
    assert (uprobs >= 0).all(), uprobs
    depth = int(np.ceil(np.log(len(uprobs)) / np.log(bfactor)))
    size = bfactor ** depth
    uprobs = np.concatenate([uprobs, np.zeros(size - len(uprobs))])
    tree = [uprobs]
    for level in reversed(range(depth - 1)):
      tree.insert(0, tree[0].reshape((-1, bfactor)).sum(-1))
    for level, prob in enumerate(tree):
      prob = prob.reshape([bfactor] * (1 + level))
      total = prob.sum(-1, keepdims=True)
      with np.errstate(divide='ignore', invalid='ignore'):
        tree[level] = np.where(total, prob / total, prob)
    return tree


class Prioritized:

  def __init__(
      self, exponent=1.0, initial=1.0, zero_on_sample=False,
      maxfrac=0.0, branching=16, seed=0):
    assert 0 <= maxfrac <= 1, maxfrac
    self.exponent = float(exponent)
    self.initial = float(initial)
    self.zero_on_sample = zero_on_sample
    self.maxfrac = maxfrac
    self.tree = SampleTree(branching, seed)
    self.prios = collections.defaultdict(lambda: self.initial)
    self.stepitems = collections.defaultdict(list)
    self.items = {}

  def prioritize(self, stepids, priorities):
    if not isinstance(stepids[0], bytes):
      stepids = [x.tobytes() for x in stepids]
    for stepid, priority in zip(stepids, priorities):
      try:
        self.prios[stepid] = priority
      except KeyError:
        print('Ignoring priority update for removed time step.')
    items = []
    for stepid in stepids:
      items += self.stepitems[stepid]
    for key in list(set(items)):
      try:
        self.tree.update(key, self._aggregate(key))
      except KeyError:
        print('Ignoring tree update for removed time step.')

  def __len__(self):
    return len(self.items)

  def __call__(self):
    key = self.tree.sample()
    if self.zero_on_sample:
      zeros = [0.0] * len(self.items[key])
      self.prioritize(self.items[key], zeros)
    return key

  def __setitem__(self, key, stepids):
    if not isinstance(stepids[0], bytes):
      stepids = [x.tobytes() for x in stepids]
    self.items[key] = stepids
    [self.stepitems[stepid].append(key) for stepid in stepids]
    self.tree.insert(key, self._aggregate(key))

  def __delitem__(self, key):
    self.tree.remove(key)
    stepids = self.items.pop(key)
    for stepid in stepids:
      stepitems = self.stepitems[stepid]
      stepitems.remove(key)
      if not stepitems:
        del self.stepitems[stepid]
        del self.prios[stepid]

  def _aggregate(self, key):
    # Both list comprehensions in this function are a performance bottleneck
    # because they are called very often.
    prios = [self.prios[stepid] for stepid in self.items[key]]
    if self.exponent != 1.0:
      prios = [x ** self.exponent for x in prios]
    mean = sum(prios) / len(prios)
    if self.maxfrac:
      return self.maxfrac * max(prios) + (1 - self.maxfrac) * mean
    else:
      return mean


class Mixture:

  def __init__(self, selectors, fractions, seed=0):
    assert set(selectors.keys()) == set(fractions.keys())
    assert sum(fractions.values()) == 1, fractions
    for key, frac in list(fractions.items()):
      if not frac:
        selectors.pop(key)
        fractions.pop(key)
    keys = sorted(selectors.keys())
    self.selectors = [selectors[key] for key in keys]
    self.fractions = np.array([fractions[key] for key in keys], np.float32)
    self.rng = np.random.default_rng(seed)

  def __call__(self):
    return self.rng.choice(self.selectors, p=self.fractions)()

  def __setitem__(self, key, stepids):
    for selector in self.selectors:
      selector[key] = stepids

  def __delitem__(self, key):
    for selector in self.selectors:
      del selector[key]

  def prioritize(self, stepids, priorities):
    for selector in self.selectors:
      if hasattr(selector, 'prioritize'):
        selector.prioritize(stepids, priorities)


class SampleTree:

  def __init__(self, branching=16, seed=0):
    assert 2 <= branching
    self.branching = branching
    self.root = SampleTreeNode()
    self.last = None
    self.entries = {}
    self.rng = np.random.default_rng(seed)

  def __len__(self):
    return len(self.entries)

  def insert(self, key, uprob):
    if not self.last:
      node = self.root
    else:
      ups = 0
      node = self.last.parent
      while node and len(node) >= self.branching:
        node = node.parent
        ups += 1
      if not node:
        node = SampleTreeNode()
        node.append(self.root)
        self.root = node
      for _ in range(ups):
        below = SampleTreeNode()
        node.append(below)
        node = below
    entry = SampleTreeEntry(key, uprob)
    node.append(entry)
    self.entries[key] = entry
    self.last = entry

  def remove(self, key):
    entry = self.entries.pop(key)
    entry_parent = entry.parent
    last_parent = self.last.parent
    entry.parent.remove(entry)
    if entry is not self.last:
      entry_parent.append(self.last)
    node = last_parent
    ups = 0
    while node.parent and not len(node):
      above = node.parent
      above.remove(node)
      node = above
      ups += 1
    if not len(node):
      self.last = None
      return
    while isinstance(node, SampleTreeNode):
      node = node.children[-1]
    self.last = node

  def update(self, key, uprob):
    entry = self.entries[key]
    entry.uprob = uprob
    entry.parent.recompute()

  def sample(self):
    node = self.root
    while isinstance(node, SampleTreeNode):
      uprobs = np.array([x.uprob for x in node.children])
      total = uprobs.sum()
      if not np.isfinite(total):
        finite = np.isinf(uprobs)
        probs = finite / finite.sum()
      elif total == 0:
        probs = np.ones(len(uprobs)) / len(uprobs)
      else:
        probs = uprobs / total
      choice = self.rng.choice(np.arange(len(uprobs)), p=probs)
      node = node.children[choice.item()]
    return node.key


class SampleTreeNode:

  __slots__ = ('parent', 'children', 'uprob')

  def __init__(self, parent=None):
    self.parent = parent
    self.children = []
    self.uprob = 0

  def __repr__(self):
    return (
        f'SampleTreeNode(uprob={self.uprob}, '
        f'children={[x.uprob for x in self.children]})'
    )

  def __len__(self):
    return len(self.children)

  def __bool__(self):
    return True

  def append(self, child):
    if child.parent:
      child.parent.remove(child)
    child.parent = self
    self.children.append(child)
    self.recompute()

  def remove(self, child):
    child.parent = None
    self.children.remove(child)
    self.recompute()

  def recompute(self):
    self.uprob = sum(x.uprob for x in self.children)
    self.parent and self.parent.recompute()


class SampleTreeEntry:

  __slots__ = ('parent', 'key', 'uprob')

  def __init__(self, key=None, uprob=None):
    self.parent = None
    self.key = key
    self.uprob = uprob
