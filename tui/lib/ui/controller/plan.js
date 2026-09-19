const { refs } = require('./state')

const getPlanScroll = () => 0

const getPlanNodes = () => {
  if (!refs.orchestrator) return []
  const plans = refs.orchestrator.plans
  if (plans.length === 0) {
    return [
      {
        type: 'planItem',
        itemClass: 'planEmpty',
        text: '  (No active plans. Model idle.)',
      },
    ]
  }

  const nodes = []
  for (const p of plans) {
    const isPlanDone = p.status === 'DONE'
    const planIcon = isPlanDone ? '✅ ' : '📋 '
    nodes.push({
      type: 'planItem',
      itemClass: isPlanDone ? 'planDone' : 'planActive',
      margin: { top: 1, bottom: 0 },
      text: `${planIcon}Plan ${p.id}: ${p.brief}`,
    })

    for (const s of p.steps) {
      let icon = '⬜ '
      let itemClass = 'planStepDefault'
      let tag = ''

      if (s.status === 'DONE') {
        icon = '✅ '
        itemClass = 'planStepDone'
      } else if (s.status === 'IN_PROGRESS') {
        icon = '▶ '
        itemClass = 'planStepActive'
        tag = ' [ACTIVE]'
      } else if (s.status === 'WAITING_FOR_USER') {
        icon = '⏳ '
        itemClass = 'planStepWaiting'
        tag = ' [WAITING]'
      } else if (s.status === 'DEFERRED') {
        icon = '⏱️ '
        itemClass = 'planStepDeferred'
        tag = ` [${s.deferReason || 'timer'}]`
      }

      nodes.push({
        type: 'planItem',
        itemClass,
        text: `   ${icon}${s.id}: ${s.brief}${tag}`,
      })
    }
  }

  return nodes
}

module.exports = {
  getPlanScroll,
  getPlanNodes,
}
