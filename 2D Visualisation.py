# 2D data plot 
fig4, axs4 = plt.subplots(2, 1)
fig4.suptitle('Four Week Consumption', fontsize=16)
plt.subplots_adjust(hspace=0.5)

for i in range(59, 83, 7):
    axs4[0].plot(data.iloc[1,i:i + 7].to_numpy(), marker='>', linestyle='-',
                 label='$week {i}$'.format(i=(i % 58) % 6))
#xs4[0].legend(loc='best')
axs4[0].set_title('With Fraud', fontsize=14)
axs4[0].set_ylabel('Consumption')
axs4[0].grid(True)

for i in range(59, 83, 7):
    axs4[1].plot(data.iloc[6,i:i + 7].to_numpy(), marker='>', linestyle='-',
                 label='$week {i}$'.format(i=(i % 58) % 6))
#xs4[1].legend(loc='best')
axs4[1].set_title('Without fraud' , fontsize=14)
axs4[1].set_ylabel('Consumption')
axs4[1].grid(True)

fig5, axs5 = plt.subplots(1, 2)
a = []
for i in range(59, 81, 7):
    a.append(data.iloc[2, i:i + 7].to_numpy())
cor = pd.DataFrame(a).transpose().corr()
cax = axs5[0].matshow(cor)
for (i, j), z in np.ndenumerate(cor):
    axs5[0].text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color='white')
alpha = ['week 1', 'week 2', 'week 3', 'week 4']
axs5[0].set_xticklabels([''] + alpha)
axs5[0].set_yticklabels([''] + alpha)
axs5[0].set_title('Customer without Fraud', fontsize=16)

a = []
for i in range(59, 83, 7):
    a.append(data.iloc[4, i:i + 7].to_numpy())
cor = pd.DataFrame(a).transpose().corr()
cax = axs5[1].matshow(cor)
for (i, j), z in np.ndenumerate(cor):
    axs5[1].text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color='white')
axs5[1].set_xticklabels([''] + alpha)
axs5[1].set_yticklabels([''] + alpha)
axs5[1].set_title('Customer with Fraud', fontsize=16)
fig5.colorbar(cax)
# plt.close('all')
plt.show()
