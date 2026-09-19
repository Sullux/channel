const formatUserTurn = (userText) => userText

const formatTurn1 = (_systemPrompt, userText) => userText

const formatUserDecisionTurn = (userText, waitingTasks = []) => {
  const waitingLines = waitingTasks
    .map((w) => `- Step ${w.id}: ${w.waitingForUser?.brief || w.brief}`)
    .join('\n')

  return [
    `User message received: "${userText}"`,
    '',
    'Tasks currently awaiting user intervention:',
    waitingLines,
    '',
    'Decision:',
    "1. If user's message fulfills an awaiting task, resume that task or complete it using the done tool.",
    '2. If user provided a new unrelated instruction, prioritize answering/planning it.',
    '3. If user cancelled a task, mark it complete/aborted.',
    'Next action:',
  ].join('\n') + '\n'
}

const formatStepTick = (planId, planBrief, stepId, stepBrief) => [
  `Focus: Plan ${planId || ''} - ${planBrief || ''}`,
  `Active Step ${stepId}: ${stepBrief}`,
  'Status: In progress.',
  'Next action:',
].join('\n') + '\n'

const formatTimerWake = (stepId, reason) => [
  `Timer expired for Step ${stepId || ''} (${reason || 'timer'}). Checking state for updates.`,
  'Next action:',
].join('\n') + '\n'

const formatResumeAfterInterrupt = (stepId, brief) => [
  `Interruption handled. Automatically resuming Step ${stepId}: ${brief}.`,
  'Previous context remains active in episodic memory.',
  'Next action:',
].join('\n') + '\n'

const formatBacklogResumeNudge = (notItem) => {
  const text = notItem.extra?.payload || notItem.preview || ''
  if (notItem.source?.startsWith('msg/user/')) {
    return `[Resume: ${notItem.id} | Source: ${notItem.source}] ${text}\nPlease continue addressing this task.`
  }
  return `[Resume Event: ${notItem.id} | Source: ${notItem.source}] ${text}\nPlease continue addressing this item.`
}

const formatNotificationInterrupt = (notItem) => {
  const text = notItem.extra?.payload || notItem.preview || ''
  if (notItem.source?.startsWith('msg/user/')) {
    return text
  }
  return `[Event: ${notItem.id} | Source: ${notItem.source}]\n${text}`
}

const formatTruncatedTurn = (userText) => userText

module.exports = {
  formatTurn1,
  formatUserTurn,
  formatUserDecisionTurn,
  formatTruncatedTurn,
  formatStepTick,
  formatTimerWake,
  formatResumeAfterInterrupt,
  formatBacklogResumeNudge,
  formatNotificationInterrupt,
}
