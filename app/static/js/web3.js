const App = {
    web3: null,
    account: null,
    meta: null,

    start: async function () {
      const {
        web3
      } = this;

      try {
        // get contract instance
        this.meta = new web3.eth.Contract(
        [{"inputs": [], "stateMutability": "nonpayable", "type": "constructor"}, {"anonymous": false, "inputs": [{"indexed": true, "internalType": "address", "name": "owner", "type": "address"}, {"indexed": true, "internalType": "address", "name": "approved", "type": "address"}, {"indexed": true, "internalType": "uint256", "name": "tokenId", "type": "uint256"}], "name": "Approval", "type": "event"}, {"anonymous": false, "inputs": [{"indexed": true, "internalType": "address", "name": "owner", "type": "address"}, {"indexed": true, "internalType": "address", "name": "operator", "type": "address"}, {"indexed": false, "internalType": "bool", "name": "approved", "type": "bool"}], "name": "ApprovalForAll", "type": "event"}, {"anonymous": false, "inputs": [{"indexed": true, "internalType": "address", "name": "from", "type": "address"}, {"indexed": true, "internalType": "address", "name": "to", "type": "address"}, {"indexed": true, "internalType": "uint256", "name": "tokenId", "type": "uint256"}], "name": "Transfer", "type": "event"}, {"inputs": [{"internalType": "address", "name": "to", "type": "address"}, {"internalType": "uint256", "name": "tokenId", "type": "uint256"}], "name": "approve", "outputs": [], "stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "address", "name": "owner", "type": "address"}], "name": "balanceOf", "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}, {"inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}], "name": "getApproved", "outputs": [{"internalType": "address", "name": "", "type": "address"}], "stateMutability": "view", "type": "function"}, {"inputs": [{"internalType": "address", "name": "owner", "type": "address"}, {"internalType": "address", "name": "operator", "type": "address"}], "name": "isApprovedForAll", "outputs": [{"internalType": "bool", "name": "", "type": "bool"}], "stateMutability": "view", "type": "function"}, {"inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "name": "modelsForSale", "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}, {"inputs": [], "name": "name", "outputs": [{"internalType": "string", "name": "", "type": "string"}], "stateMutability": "view", "type": "function"}, {"inputs": [{"internalType": "string", "name": "", "type": "string"}], "name": "nameToModel", "outputs": [{"internalType": "string", "name": "name", "type": "string"}, {"internalType": "uint256", "name": "accuracy", "type": "uint256"}, {"internalType": "string", "name": "description", "type": "string"}, {"internalType": "uint256", "name": "tokenID", "type": "uint256"}], "stateMutability": "view", "type": "function"}, {"inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}], "name": "ownerOf", "outputs": [{"internalType": "address", "name": "", "type": "address"}], "stateMutability": "view", "type": "function"}, {"inputs": [{"internalType": "address", "name": "from", "type": "address"}, {"internalType": "address", "name": "to", "type": "address"}, {"internalType": "uint256", "name": "tokenId", "type": "uint256"}], "name": "safeTransferFrom", "outputs": [], "stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "address", "name": "from", "type": "address"}, {"internalType": "address", "name": "to", "type": "address"}, {"internalType": "uint256", "name": "tokenId", "type": "uint256"}, {"internalType": "bytes", "name": "_data", "type": "bytes"}], "name": "safeTransferFrom", "outputs": [], "stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "address", "name": "operator", "type": "address"}, {"internalType": "bool", "name": "approved", "type": "bool"}], "name": "setApprovalForAll", "outputs": [], "stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "bytes4", "name": "interfaceId", "type": "bytes4"}], "name": "supportsInterface", "outputs": [{"internalType": "bool", "name": "", "type": "bool"}], "stateMutability": "view", "type": "function"}, {"inputs": [], "name": "symbol", "outputs": [{"internalType": "string", "name": "", "type": "string"}], "stateMutability": "view", "type": "function"}, {"inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "name": "tokenIdToModel", "outputs": [{"internalType": "string", "name": "name", "type": "string"}, {"internalType": "uint256", "name": "accuracy", "type": "uint256"}, {"internalType": "string", "name": "description", "type": "string"}, {"internalType": "uint256", "name": "tokenID", "type": "uint256"}], "stateMutability": "view", "type": "function"}, {"inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}], "name": "tokenURI", "outputs": [{"internalType": "string", "name": "", "type": "string"}], "stateMutability": "view", "type": "function"}, {"inputs": [{"internalType": "address", "name": "from", "type": "address"}, {"internalType": "address", "name": "to", "type": "address"}, {"internalType": "uint256", "name": "tokenId", "type": "uint256"}], "name": "transferFrom", "outputs": [], "stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "string", "name": "_name", "type": "string"}, {"internalType": "uint256", "name": "_accuracy", "type": "uint256"}, {"internalType": "string", "name": "_description", "type": "string"}], "name": "createModel", "outputs": [], "stateMutability": "payable", "type": "function"}, {"inputs": [{"internalType": "string", "name": "_name", "type": "string"}, {"internalType": "uint256", "name": "_price", "type": "uint256"}], "name": "putModelUpForSale", "outputs": [], "stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "string", "name": "_name", "type": "string"}], "name": "buyModel", "outputs": [], "stateMutability": "payable", "type": "function"}, {"inputs": [{"internalType": "string", "name": "_name", "type": "string"}], "name": "checkPrice", "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}],
          '0x85De58930C8955eB3F35e1A3b8160d5B69671564',
        );

        // get accounts
        const accounts = await web3.eth.getAccounts();
        this.account = accounts[0];
      } catch (error) {
        console.error("Could not connect to contract or chain.");
      }
    },

    createModel: async function () {
      const {
        createModel
      } = this.meta.methods;
      document.getElementById('mint-message').innerHTML = 'Minting your AI model...';
      document.getElementById('mintBTN').style.display = 'none';
      document.getElementById('mint-progress').style.display = 'initial';

      const name = document.getElementById("modelName").value;
      const accuracy = document.getElementById("modelAccuracy").value;
      const description = document.getElementById("modelDescription").value;
      const etherPrice = 0.015;
      const weiPrice = this.web3.utils.toWei(etherPrice.toString(), 'ether');
      await createModel(name, accuracy, description).send({
        from: this.account,
        value: weiPrice
      });
      $.post("/mint", {
        data: this.account,
        type: document.getElementById('modelType').value,
        name: document.getElementById('modelName').value
      }, function(err, req, resp) {
        document.getElementById('mint-progress').style.display = 'none';
        document.getElementById('mintBTN').style.display = 'intial';
        window.location.href = "/profile"
      });
    },

    putModelUpForSale: async function () {
      const {
        putModelUpForSale
      } = this.meta.methods;
      document.getElementById('listBTN').style.display = 'none';
      document.getElementById('list-progress').style.display = 'initial';

      const name = document.getElementById('sell-name').value;
      const price = Number(document.getElementById('sell-price').value);
      const weiPrice = this.web3.utils.toWei(price.toString(), 'ether');
      await putModelUpForSale(name, weiPrice).send({
        from: this.account
      });
      $.post("/list_item", {
        name: name,
        price: price
      }, function (err, req, resp) {
        document.getElementById('list-progress').style.display = 'none';
        document.getElementById('listBTN').style.display = 'intial';
        window.location.href = "/profile"
      });
    },
};

window.App = App;

const enableBTN = document.getElementById('enable');
enableBTN.addEventListener("click", async function () {
  if (window.ethereum) {
    // use MetaMask's provider
    App.web3 = new Web3(window.ethereum);
    accounts = await ethereum.request({method: 'eth_requestAccounts'}); // get permission to access accounts
    App.start();
    $.post("/login_post", {
      data: accounts[0]
    }, function(err, req, resp){
      window.location.href = "/profile"
    });
  } else {
    alert('No Ethereum Wallet Providers Detected. Also, mobile devices are currently not supported.')
  };
});


window.ethereum.on('accountsChanged', async function (accounts) {
  accounts = await ethereum.request({method: 'eth_requestAccounts'}); // get permission to access accounts
  $.post("/login_post", {
    data: accounts[0]
  }, function(err, req, resp){
    var user = document.getElementById('user');
    window.location.href = "/profile"
    user.innerHTML = resp
  });
});

window.ethereum.on('chainChanged', (chainId) => {
  // Handle the new chain.
  const chainIDs = {'Mainnet': '0x1',
                    'Ropsten': '0x3',
                    'Rinkeby': '0x4',
                    'Goerli': '0x5',
                    'Kovan': '0x2a'};
  var hasVal = Object.values(chainIDs).includes(chainId);
  if (hasVal) {
    var idx = Object.values(chainIDs).indexOf(chainId);
    var network = Object.keys(chainIDs)[idx];

    var message = `You have changed to the ${network} Ethereum Network`;
    alert(message);
    window.location.reload();
  } else {
    var message = 'You have changed to an undefined Ethereum network';
    alert(message);
    window.location.reload();
  };
});